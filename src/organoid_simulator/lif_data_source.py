"""
Leaky Integrate-and-Fire (LIF) simulator data source for the CL SDK.

Mirrored into Benchmarking/ from cl1-callosum-biocompute/organoid_simulator so
this folder is self-contained (see Benchmarking/README.md). This copy has since
diverged from the original: it adds the ``n_units`` "spatial mode" described
below, used by exp7 to decouple neuron count from electrode count. The
original 64-neuron, one-per-channel model is unchanged and is still the
default.

This is a drop-in replacement for the SDK's built-in ``RandomDataSource`` that,
unlike the random source, *reacts to stimulation*. Stimulating a channel injects
current into that channel's model neuron, which can drive it across threshold and
produce a genuine, timed, evoked spike -- so you can sandbox closed-loop
stimulus->response hypotheses on a laptop before spending money on real CL1
hardware.

Register it exactly like any other simulator source (no changes to your loop code)::

    import cl
    from organoid_simulator import make_lif_source

    cl.sim.set_simulator_data_source(
        "organoid_simulator.lif_data_source:make_lif_source",
        config={"random_seed": 42},
    )

    with cl.open() as neurons:
        for tick in neurons.loop(ticks_per_second=50):
            neurons.stim(cl.ChannelSet(10), cl.StimDesign(200, -1.5, 200, 1.5))
            spikes = tick.analysis.spikes   # now reacts to the stim above

Scope (deliberately the *minimal* viable dynamics -- see the repo plan file):

* 64 independent LIF neurons by default, one per MEA channel. Pass ``n_units``
  to decouple neuron count from electrode count instead: n_units neurons are
  scattered at continuous positions over the same 8x8 footprint, and each
  electrode reads a distance-weighted mixture of nearby neurons -- closer to
  what a real MEA sees than a clean one neuron/one channel wire. See
  ``LIFDataSource``'s docstring for the "spatial mode" parameters.
* Two opt-in knobs, both off by default: ``coupling_strength`` (flat 4-neighbour
  grid coupling, enough to see activity propagate) and ``stim_artifact`` (a crude
  amplifier-saturation bump on the raw trace). In spatial mode, coupling runs
  over a sparse spatial small-world graph between neurons instead of the dense
  64x64 channel grid.
* No synaptic plasticity, no E/I cell types, no true amplifier blanking, and a
  linear (not strength-duration) charge model. Those are future scope.
  Plasticity/STDP are additionally unimplemented in spatial mode specifically
  (see ``LIFDataSource.__init__``).

Two modelling decisions worth calling out because they are *not* textbook LIF, and
are made on purpose:

1. **The recorded frames are extracellular, not the membrane voltage.** A real MEA
   records a small (tens-of-uV) noisy echo of nearby transmembrane currents, not
   the ~100 mV intracellular swing. So the membrane voltage ``V`` is internal state
   only and *never* written to ``frames``. Frames are built as a noise floor plus a
   stamped canonical extracellular spike waveform. (Writing ``V`` into an int16
   frame -- as a naive sketch would -- overflows the +-32767 range by ~15x and
   represents the wrong physical quantity entirely.)

2. **Only the cathodic (depolarising) phase of a stim excites.** CL stim pulses are
   charge-balanced (the anodic recharge phase exactly cancels the cathodic charge),
   so a purely linear integrator would net ~zero drive and never fire. Physically,
   the leading cathodic phase is the excitatory one and an all-or-none threshold
   crossing during it is not undone by the later recharge. We model that by drawing
   excitatory drive from the cathodic (negative-current, by CL convention) part of
   the waveform only.

All internal "gains" are expressed directly in **mV of membrane drive per frame**,
so there is no MOhm/uA unit juggling (and no unit bug: MOhm x uA is volts, not mV).
Frames are integrated one at a time because the threshold-and-reset nonlinearity
can't be vectorised over time; expect roughly 10-50x real time in accelerated mode
on an Apple-silicon laptop, which is ample for closed-loop iteration.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.sparse import csr_matrix

from cl.sim import (
    DataSourceBatch,
    DataSourceSpike,
    DataSourceStim,
    SimulatorDataSource,
    SimulatorDataSourceMetadata,
)

# --- Fixed SDK contract values (25 kHz, 64ch, extracellular spike snippet shape).
# Hardcoded rather than imported from cl._sim._data_buffer (a private module) since
# these are stable parts of the on-device contract.
FRAMES_PER_SECOND    = 25_000
CHANNEL_COUNT        = 64
US_PER_FRAME         = 1_000_000 / FRAMES_PER_SECOND          # 40 us
MS_PER_FRAME         = 1_000 / FRAMES_PER_SECOND              # 0.04 ms
UV_PER_SAMPLE_UNIT   = 0.195                                  # raw int16 unit -> uV
SPIKE_SAMPLES_BEFORE = 25
SPIKE_SAMPLES_TOTAL  = 75                                     # 25 before + peak + 49 after
INT16_MAX            = 32_767
INT16_MIN            = -32_768

# 8x8 MEA layout. Corners are grounded, channel 4 is the reference; none of these
# ever spike (matches RandomDataSource so behaviour is consistent with the default).
GROUNDED_CHANNELS  = (0, 7, 56, 63)
REFERENCE_CHANNELS = (4,)
NON_SPIKING        = frozenset(GROUNDED_CHANNELS + REFERENCE_CHANNELS)


# --------------------------------------------------------------------------- #
# Connectivity graphs. A connectivity matrix W is (64, 64) where W[i, j] is the
# weight of the directed edge from presynaptic channel j to postsynaptic channel
# i (so the coupling drive to i is sum_j W[i, j] * fired[j]). Grounded/reference
# channels and self-edges are always zeroed. Any (64, 64) array can be passed to
# LIFDataSource(connectivity=...), so you can plug in your own graph.
# --------------------------------------------------------------------------- #

def _grid_pos(channel: int) -> tuple[int, int]:
    """(row, col) of a channel on the 8x8 array (channel = row + 8*col)."""
    return channel % 8, channel // 8


def _electrode_positions() -> np.ndarray:
    """(CHANNEL_COUNT, 2) array of every electrode's (row, col), integer
    electrode-pitch coordinates. Used to place and project spatial-mode
    neurons against the real 8x8 footprint."""
    return np.array([_grid_pos(c) for c in range(CHANNEL_COUNT)], dtype=float)


def grid_connectivity(non_spiking=NON_SPIKING) -> np.ndarray:
    """The original unweighted 4-nearest-neighbour grid (all edges weight 1).

    Regular and homogeneous: every neuron couples equally to its up/down/left/
    right neighbours. Used by the reproducible demos so their ground-truth
    structure is exact.
    """
    W = np.zeros((CHANNEL_COUNT, CHANNEL_COUNT))
    for ch in range(CHANNEL_COUNT):
        r, c = _grid_pos(ch)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 8 and 0 <= cc < 8:
                W[ch, rr + 8 * cc] = 1.0
    W[list(non_spiking), :] = 0.0
    W[:, list(non_spiking)] = 0.0
    np.fill_diagonal(W, 0.0)
    return W


def small_world_connectivity(
    k               : int   = 8,
    rewire_p        : float = 0.15,
    weight_sigma    : float = 0.5,
    seed            : int   = 0,
    target_in_weight: float = 4.0,
    non_spiking     = NON_SPIKING,
) -> np.ndarray:
    """A weighted, spatially-embedded small-world connectivity matrix.

    Construction (Watts-Strogatz style, embedded in the 8x8 array):
      1. Each neuron receives from its ``k`` spatially-nearest neighbours -- a
         dense local lattice with high clustering (many triangles).
      2. Each such edge is rewired with probability ``rewire_p`` to a random
         source anywhere on the array, adding a few long-range shortcuts that
         collapse the path length -> the small-world regime (high clustering +
         short paths).
      3. Edge weights are lognormal (heterogeneous, a few strong pathways), and
         the whole matrix is scaled so the mean total incoming weight equals
         ``target_in_weight`` (the grid's ~4), keeping ``coupling_gain_mV``
         interpretable across graphs.

    Signals therefore mostly diffuse locally but occasionally jump across the
    array -- closer to a real culture than uniform grid diffusion.
    """
    rng  = np.random.default_rng(seed)
    live = [c for c in range(CHANNEL_COUNT) if c not in non_spiking]
    pos  = np.array([_grid_pos(c) for c in range(CHANNEL_COUNT)], dtype=float)
    W    = np.zeros((CHANNEL_COUNT, CHANNEL_COUNT))

    for i in live:                                  # postsynaptic neuron i
        dist = np.hypot(pos[:, 0] - pos[i, 0], pos[:, 1] - pos[i, 1])
        dist[i] = np.inf
        dist[list(non_spiking)] = np.inf
        sources = np.argsort(dist)[:k]              # k nearest presynaptic sources
        for j in sources:
            src = int(j)
            if rng.random() < rewire_p:             # rewire -> long-range shortcut
                src = int(rng.choice(live))
            if src == i:
                continue
            W[i, src] = rng.lognormal(mean=0.0, sigma=weight_sigma)

    row_sums = W.sum(axis=1)
    mean_in  = row_sums[row_sums > 0].mean()
    if mean_in > 0:
        W *= target_in_weight / mean_in
    W[list(non_spiking), :] = 0.0
    W[:, list(non_spiking)] = 0.0
    np.fill_diagonal(W, 0.0)
    return W


# --------------------------------------------------------------------------- #
# Spatial mode (n_units != CHANNEL_COUNT): many neurons embedded at continuous
# positions over the same 8x8 electrode footprint, each electrode picking up a
# distance-weighted mixture of nearby neurons -- the same "electrode sees a
# projection, not a unit" model used in Benchmarking/analysis/exp1_detection.py
# and lif_groundtruth.py, ported into the live SDK-registered source so exp7
# can compare it against a real replay through the identical read path.
# --------------------------------------------------------------------------- #

def _generate_unit_positions(n_units: int, seed: int, pad: float) -> np.ndarray:
    """n_units positions (row, col) uniform over the padded 8x8 footprint.

    pad is in electrode-pitch units (electrodes sit at integer coordinates
    0..7). A small pad keeps every unit within reach of at least its nearest
    electrode once the projection cutoff below is applied.
    """
    rng = np.random.default_rng(seed)
    lo, hi = -pad, 7.0 + pad
    return rng.uniform(lo, hi, size=(n_units, 2))


def _build_projection(
    unit_pos: np.ndarray,
    elec_pos: np.ndarray,
    peak_uv: np.ndarray,
    sigma_pitch: float,
    cutoff_sigma: float,
    non_spiking,
) -> np.ndarray:
    """(n_units, CHANNEL_COUNT) amplitude matrix: how many uV each unit's spike
    contributes to each electrode. Exponential falloff with distance in
    electrode-pitch units (matches exp1's peak_uv * exp(-d/108um) on a 200um
    pitch: 108/200 ~= 0.5, the default sigma_pitch here), hard-cut beyond
    cutoff_sigma * sigma_pitch so the projection stays sparse.
    """
    d = np.sqrt(((unit_pos[:, None, :] - elec_pos[None, :, :]) ** 2).sum(axis=2))
    A = peak_uv[:, None] * np.exp(-d / sigma_pitch)
    A[d > cutoff_sigma * sigma_pitch] = 0.0
    A[:, list(non_spiking)] = 0.0
    return A


def spatial_small_world_sparse(
    positions: np.ndarray,
    k: int = 8,
    rewire_p: float = 0.15,
    weight_sigma: float = 0.5,
    seed: int = 0,
    target_in_weight: float = 4.0,
) -> csr_matrix:
    """Unit-to-unit small-world coupling graph over continuous positions, the
    same construction as :func:`small_world_connectivity` generalised beyond
    the 64-channel grid: each unit receives from its k spatially-nearest
    neighbours, each edge rewired with probability rewire_p to a random unit,
    lognormal weights scaled so mean incoming weight equals target_in_weight.

    Returned sparse (not dense) because at n_units in the thousands a dense
    (n_units, n_units) coupling matrix is fine on memory but a dense matmul
    every 40us frame is not -- a k-sparse matrix-vector product is.
    """
    n = positions.shape[0]
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        dist = np.hypot(positions[:, 0] - positions[i, 0], positions[:, 1] - positions[i, 1])
        dist[i] = np.inf
        sources = np.argpartition(dist, min(k, n - 1) - 1)[:k]
        for j in sources:
            src = int(j)
            if rng.random() < rewire_p:
                src = int(rng.integers(n))
            if src == i:
                continue
            rows.append(i)
            cols.append(src)
            vals.append(rng.lognormal(mean=0.0, sigma=weight_sigma))
    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    mean_in = row_sums[row_sums > 0].mean() if (row_sums > 0).any() else 1.0
    if mean_in > 0:
        W = W.multiply(target_in_weight / mean_in).tocsr()
    return W


class LIFDataSource(SimulatorDataSource):
    """An LIF network that reacts to CL stimulation. See module docstring.

    Two modes, chosen by ``n_units``:

    * ``n_units == 64`` (default) -- one neuron per electrode, exactly the
      original 1:1 model.
    * ``n_units != 64`` -- "spatial mode". ``n_units`` neurons are scattered
      at continuous positions over the same 8x8 footprint; each of the 64
      electrodes reads a distance-weighted mixture of nearby neurons (see
      :func:`_build_projection`), and stimulating an electrode drives nearby
      neurons rather than one neuron 1:1. Coupling, if enabled, runs over a
      sparse spatial small-world graph between neurons rather than the dense
      64x64 channel grid. Plasticity/STDP are not implemented in this mode.
    """

    def __init__(
        self,
        random_seed              : int | None = None,
        tau_mem_ms               : float = 20.0,
        v_rest_mV                : float = -70.0,
        v_thresh_mV              : float = -50.0,
        v_reset_mV               : float = -75.0,
        refractory_ms            : float = 2.0,
        background_drive_std_mV  : float = 0.60,
        stim_gain_mV_per_uA      : float = 5.0,
        coupling_gain_mV         : float = 0.0,
        spike_amplitude_uV       : float = 80.0,
        noise_floor_uV           : float = 6.0,
        stim_artifact            : bool  = False,
        stim_artifact_gain_uV_per_uA: float = 4000.0,
        plasticity               : bool  = False,
        plasticity_lr            : float = 0.02,
        eligibility_tau_ms       : float = 200.0,
        weight_max_mV            : float = 6.0,
        stdp                     : bool  = False,
        stdp_tau_pre_ms          : float = 20.0,
        stdp_tau_post_ms         : float = 20.0,
        stdp_ltd_ratio           : float = 0.5,
        connectivity             : np.ndarray | None = None,
        small_world_k            : int   = 8,
        small_world_rewire_p     : float = 0.15,
        small_world_weight_sigma : float = 0.5,
        connectivity_seed        : int   = 0,
        n_units                  : int   = CHANNEL_COUNT,
        unit_position_seed       : int | None = None,
        unit_position_pad        : float = 0.4,
        projection_sigma_pitch   : float = 0.5,
        projection_cutoff_sigma  : float = 3.0,
        spike_amplitude_lognorm_sigma: float = 0.55,
        force_spatial             : bool | None = None,
        separate_noise_streams     : bool = False,
    ):
        """
        Args:
            random_seed: Seed for the internal RNG (background drive + noise floor).
                Falls back to ``CL_SDK_RANDOM_SEED`` then a fresh random seed.
            tau_mem_ms: Membrane time constant (ms).
            v_rest_mV / v_thresh_mV / v_reset_mV: Resting / firing-threshold /
                post-spike reset potentials (mV). Threshold minus rest sets how much
                depolarising drive is needed to fire.
            refractory_ms: Absolute refractory period after a spike (ms).
            background_drive_std_mV: Std of the per-frame stochastic membrane drive
                (mV). Larger -> higher spontaneous firing rate. Set to 0 for a silent
                culture. Default gives a low, non-zero baseline rate.
            stim_gain_mV_per_uA: Membrane depolarisation per uA of *cathodic* stim
                current, per frame of the pulse. Larger -> weaker stimuli fire.
            coupling_gain_mV: OPT-IN. Depolarisation added to a neuron per spiking
                4-neighbour on the previous frame. 0 disables coupling (independent
                neurons). A few mV lets activity propagate across the grid.
            spike_amplitude_uV: Peak-to-trough scale of the canonical extracellular
                spike waveform stamped into frames and returned as ``spike.samples``.
            noise_floor_uV: Std of the per-channel recording noise floor (uV).
            stim_artifact: OPT-IN. If True, add a large transient to the raw trace on
                stimulated channels during the pulse (cosmetic; does NOT blank spike
                detection the way real hardware does).
            stim_artifact_gain_uV_per_uA: Scale of that artifact bump.
            plasticity: OPT-IN reward-modulated Hebbian plasticity on the coupling
                weights (default off -> the v1 contract of a fixed substrate holds).
                When on, co-active pre->post pairs leave an eligibility trace that a
                later ``apply_reward(r)`` consolidates (r>0) or depresses (r<0). This
                is a simple three-factor rule for closed-loop / free-energy style
                experiments, not a validated model of biological learning. Implies
                coupling even if ``coupling_gain_mV`` is 0 (weights start at the
                nearest-neighbour structure and may grow elsewhere as pathways form).
            plasticity_lr: Weight change per unit reward per unit eligibility.
            eligibility_tau_ms: Decay time constant of the co-activity trace (ms).
            weight_max_mV: Upper clip on any learned coupling weight.
            stdp: OPT-IN. Replace the 1-frame (40 us) pre->post coincidence rule
                with a trace-based STDP rule. Each neuron keeps a decaying
                presynaptic trace; when a neuron fires it credits synapses from
                *recently* active presynaptic partners, so associations across
                millisecond-scale gaps can form (needed for spatio-temporal
                learning). Default off keeps the original instantaneous rule so
                existing plasticity users are unchanged.
            stdp_tau_pre_ms / stdp_tau_post_ms: Decay time constants of the pre/
                post traces -- these set the temporal pairing window.
            stdp_ltd_ratio: Strength of acausal (post-before-pre) depression
                relative to causal (pre-before-post) potentiation.
            connectivity: Structural coupling graph as a (64, 64) matrix where
                W[i, j] is the weight from presynaptic j to postsynaptic i. Pass
                your own graph here (e.g. from metrics.graph_calculations or any
                generator). If None (default), a weighted small-world graph is
                generated -- see small_world_connectivity(). The actual coupling
                strength is this matrix times coupling_gain_mV.
            small_world_k / small_world_rewire_p / small_world_weight_sigma /
            connectivity_seed: Parameters of the default small-world generator
                (ignored if `connectivity` is given).
            n_units: Number of model neurons. Default CHANNEL_COUNT (64) keeps
                the original one-neuron-per-electrode model. Any other value
                switches to "spatial mode": n_units neurons are scattered over
                the same 8x8 electrode footprint and each electrode reads a
                distance-weighted mixture of nearby neurons -- see the class
                docstring. Not compatible with plasticity/stdp.
            unit_position_seed: Seed for neuron placement in spatial mode.
                Falls back to connectivity_seed.
            unit_position_pad: How far outside the 0..7 electrode footprint
                neurons may be placed (electrode-pitch units). Kept small so
                every neuron stays within projection_cutoff_sigma of its
                nearest electrode.
            projection_sigma_pitch: Exponential falloff length constant for
                the electrode projection, in electrode-pitch units (default
                0.5, matching exp1_detection.py's 108um decay on a 200um
                pitch). Smaller -> each neuron is seen by fewer, sharper
                electrodes.
            projection_cutoff_sigma: Hard cutoff radius for the projection,
                in multiples of projection_sigma_pitch. Keeps the projection
                (and therefore the per-frame stamping cost) sparse.
            spike_amplitude_lognorm_sigma: Log-space spread of each neuron's
                peak amplitude around spike_amplitude_uV, in spatial mode
                (matches exp1_detection.py's lognormal amplitude model).
            separate_noise_streams: Keep background drive and recording noise
                on independent seeded streams, so read batch boundaries do not
                change the generated activity. Default False preserves old runs.
        """
        # NB: __init__ runs in BOTH the parent and the simulator subprocess. Keep it
        # to storing plain config; all mutable simulation state is created in open().
        self._random_seed          = random_seed
        self._tau_mem_ms           = float(tau_mem_ms)
        self._v_rest               = float(v_rest_mV)
        self._v_thresh             = float(v_thresh_mV)
        self._v_reset              = float(v_reset_mV)
        self._refractory_frames    = max(1, round(refractory_ms / MS_PER_FRAME))
        self._bg_std_mV            = float(background_drive_std_mV)
        self._stim_gain            = float(stim_gain_mV_per_uA)
        self._coupling_gain        = float(coupling_gain_mV)
        self._spike_amplitude_uV   = float(spike_amplitude_uV)
        self._noise_floor_uV       = float(noise_floor_uV)
        self._stim_artifact        = bool(stim_artifact)
        self._artifact_gain        = float(stim_artifact_gain_uV_per_uA)

        self._plastic              = bool(plasticity)
        self._plasticity_lr        = float(plasticity_lr)
        self._weight_max           = float(weight_max_mV)
        # Coupling is active if it was requested directly OR plasticity can grow it.
        self._use_coupling         = (self._coupling_gain > 0.0) or self._plastic

        self._stdp       = bool(stdp)
        self._ltd_ratio  = float(stdp_ltd_ratio)
        self._pre_decay  = float(np.exp(-MS_PER_FRAME / stdp_tau_pre_ms))
        self._post_decay = float(np.exp(-MS_PER_FRAME / stdp_tau_post_ms))

        self._connectivity_arg    = None if connectivity is None else np.asarray(connectivity, dtype=np.float64)
        self._sw_k                = int(small_world_k)
        self._sw_rewire_p         = float(small_world_rewire_p)
        self._sw_weight_sigma     = float(small_world_weight_sigma)
        self._connectivity_seed   = int(connectivity_seed)

        self._leak_coeff = MS_PER_FRAME / self._tau_mem_ms   # dt/tau per Euler step
        self._elig_decay = float(np.exp(-MS_PER_FRAME / eligibility_tau_ms))

        # -- spatial mode (n_units != CHANNEL_COUNT) --------------------------
        self._n_units          = int(n_units)
        self._spatial          = (self._n_units != CHANNEL_COUNT) if force_spatial is None else bool(force_spatial)
        self._pos_seed         = connectivity_seed if unit_position_seed is None else int(unit_position_seed)
        self._pos_pad          = float(unit_position_pad)
        self._proj_sigma       = float(projection_sigma_pitch)
        self._proj_cutoff      = float(projection_cutoff_sigma)
        self._amp_lognorm_sig  = float(spike_amplitude_lognorm_sigma)
        if self._spatial and (self._plastic or self._stdp):
            raise NotImplementedError(
                "plasticity/stdp are not implemented in spatial mode (n_units "
                f"= {self._n_units} != {CHANNEL_COUNT}); the eligibility trace "
                "is an (n_units, n_units) outer product per frame, which is "
                "only tractable at channel-count scale."
            )

        # Channels allowed to spike (everything except grounded + reference).
        # In spatial mode this instead gates which ELECTRODES may register a
        # projected spike; neurons themselves have no such restriction.
        self._spiking_mask = np.ones(CHANNEL_COUNT, dtype=bool)
        self._spiking_mask[list(NON_SPIKING)] = False

        self._metadata = SimulatorDataSourceMetadata(
            channel_count        = CHANNEL_COUNT,
            frames_per_second    = FRAMES_PER_SECOND,
            uV_per_sample_unit   = UV_PER_SAMPLE_UNIT,
            start_timestamp      = 0,
            duration_frames      = None,   # unbounded
            seekable             = False,  # state depends on full stim history
            supports_accelerated = True,   # pure numpy, no wall-clock dependency
        )

        # Simulation state (populated in open()).
        self._rng               : np.random.Generator | None = None
        self._separate_noise_streams = bool(separate_noise_streams)
        self._drive_rng         : np.random.Generator | None = None
        self._noise_rng         : np.random.Generator | None = None
        self._V                 : np.ndarray | None = None
        self._refractory_until  : np.ndarray | None = None
        self._fired_prev        : np.ndarray | None = None
        self._adjacency         : np.ndarray | None = None
        self._spike_template    : np.ndarray | None = None
        self._W                 : np.ndarray | None = None   # coupling weights (mV)
        self._elig              : np.ndarray | None = None   # co-activity eligibility trace
        self._x_pre             : np.ndarray | None = None   # presynaptic STDP trace
        self._y_post            : np.ndarray | None = None   # postsynaptic STDP trace
        # -- spatial-mode-only state --
        self._unit_pos          : np.ndarray | None = None   # (n_units, 2)
        self._projection        : np.ndarray | None = None   # (n_units, 64) uV
        self._unit_channels     : list | None = None         # per-unit [(ch, amp), ...]
        self._home_channel      : np.ndarray | None = None   # (n_units,) int
        self._stim_spread       : np.ndarray | None = None   # (n_units, 64) spatial kernel
        # Future contributions keyed by absolute frame timestamp. Persist across
        # read() calls because a pulse / spike waveform can span into the next batch.
        self._pending_current   : dict[int, np.ndarray] = {}   # ts -> (64,) uA (signed)
        self._pending_waveform  : dict[int, np.ndarray] = {}   # ts -> (64,) uV additive
        self._next_read_ts      : int = 0                      # first frame the next read() produces

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return self._metadata

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        seed = self._random_seed
        if seed is None:
            env = os.getenv("CL_SDK_RANDOM_SEED")
            seed = int(env) if env else int(np.random.SeedSequence(None).generate_state(1)[0])
        self._rng = np.random.default_rng(seed)
        if self._separate_noise_streams:
            drive_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
            self._drive_rng = np.random.default_rng(drive_seed)
            self._noise_rng = np.random.default_rng(noise_seed)
        else:
            self._drive_rng = self._noise_rng = self._rng

        n = self._n_units
        self._V                = np.full(n, self._v_rest, dtype=np.float64)
        self._refractory_until = np.full(n, -1, dtype=np.int64)
        self._fired_prev       = np.zeros(n, dtype=np.float64)
        self._spike_template   = self._build_spike_template()

        if self._spatial:
            self._setup_spatial()
            # Sparse k-neighbour graph; gain-scaled the same way as the dense
            # 1:1 case. Plasticity is disallowed in spatial mode (see
            # __init__), so there's no eligibility/STDP state to allocate.
            self._adjacency = spatial_small_world_sparse(
                self._unit_pos, k=self._sw_k, rewire_p=self._sw_rewire_p,
                weight_sigma=self._sw_weight_sigma, seed=self._connectivity_seed,
            )
            self._W = (self._coupling_gain * self._adjacency).tocsr()
            self._elig = self._x_pre = self._y_post = None
        else:
            self._adjacency = self._resolve_connectivity()
            # Coupling weights start at the (gain-scaled) nearest-neighbour
            # structure. With plasticity on, learning may grow weights off
            # this initial pattern.
            self._W      = self._coupling_gain * self._adjacency.copy()
            self._elig   = np.zeros((CHANNEL_COUNT, CHANNEL_COUNT), dtype=np.float64)
            self._x_pre  = np.zeros(CHANNEL_COUNT, dtype=np.float64)
            self._y_post = np.zeros(CHANNEL_COUNT, dtype=np.float64)

        self._pending_current.clear()
        self._pending_waveform.clear()
        self._next_read_ts = 0

        mode =f"spatial ({n} neurons -> {CHANNEL_COUNT} electrodes)" if self._spatial else "1:1 (64 neurons)"
        print(f"LIF data source is using seed: {seed}  [{mode}]")

    def close(self) -> None:
        self._rng = None
        self._drive_rng = None
        self._noise_rng = None
        self._V = None
        self._refractory_until = None
        self._fired_prev = None
        self._pending_current.clear()
        self._pending_waveform.clear()

    # -- stim handling -------------------------------------------------------

    def on_stims(self, stims):
        """Expand committed stim pulses into per-frame current, keyed by timestamp.

        Delivered by the producer *before* read() for the overlapping window, so a
        stim can drive a spike in the very batch it lands in. Storing current at the
        exact frame timestamp (rather than bumping "now") preserves stim->spike
        latency instead of quantising it to a whole tick.

        In real time the producer can already have generated a stim's frames by
        the time the stim arrives. Such a stim is delivered at the first frame
        still to be read rather than dropped as stale.
        """
        for stim in stims:
            start_ts     = max(int(stim.timestamp), self._next_read_ts)
            frame_offset = 0
            for duration_us, current_uA in zip(stim.phase_durations_us, stim.phase_currents_uA):
                # A 20 us phase is shorter than one 40 us frame; never drop it.
                n_frames = max(1, round(duration_us / US_PER_FRAME))
                for k in range(n_frames):
                    ts  = start_ts + frame_offset + k
                    vec = self._pending_current.get(ts)
                    if vec is None:
                        vec = np.zeros(CHANNEL_COUNT, dtype=np.float64)
                        self._pending_current[ts] = vec
                    vec[stim.channel] += float(current_uA)
                frame_offset += n_frames

    # -- read loop -----------------------------------------------------------

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        assert self._V is not None and self._drive_rng is not None and self._noise_rng is not None, "read() before open()"
        to_timestamp = from_timestamp + frame_count

        self._drop_stale_pending(from_timestamp)

        # Vectorise what we can (noise is independent per frame); the V update itself
        # must stay sequential because of the threshold-and-reset nonlinearity.
        # bg_drive is per NEURON (n_units in spatial mode, else per channel);
        # noise_floor is always per ELECTRODE -- it's recording noise, not neuron
        # activity, so it never changes shape between modes.
        bg_drive    = self._drive_rng.normal(0.0, self._bg_std_mV, size=(frame_count, self._n_units))
        noise_floor = self._noise_rng.normal(0.0, self._noise_floor_uV, size=(frame_count, CHANNEL_COUNT))

        frames_uV = noise_floor
        spikes: list[DataSourceSpike] = []

        V              = self._V
        refr_until     = self._refractory_until
        fired_prev     = self._fired_prev
        W              = self._W
        elig           = self._elig
        x_pre          = self._x_pre
        y_post         = self._y_post
        use_coupling   = self._use_coupling
        plastic        = self._plastic
        stdp           = self._stdp
        pre_decay      = self._pre_decay
        post_decay     = self._post_decay
        ltd_ratio      = self._ltd_ratio
        elig_decay     = self._elig_decay
        leak_coeff     = self._leak_coeff
        v_rest         = self._v_rest
        stim_gain      = self._stim_gain
        spatial        = self._spatial
        stim_spread    = self._stim_spread   # (n_units, 64), spatial mode only

        for step in range(frame_count):
            ts = from_timestamp + step

            # --- membrane update (all terms in mV) -------------------------
            drive = bg_drive[step]
            I_ext = self._pending_current.pop(ts, None)
            if I_ext is not None:
                # Cathodic (negative, by CL convention) current excites; the anodic
                # recharge phase is intentionally ignored (see module docstring).
                cathodic = np.maximum(0.0, -I_ext)             # (64,), per electrode
                if spatial:
                    # Spread each stimulated electrode's field to nearby neurons
                    # rather than driving a single 1:1 neuron.
                    drive = drive + stim_gain * (stim_spread @ cathodic)
                else:
                    drive = drive + stim_gain * cathodic
            if use_coupling:
                if spatial:
                    # Sparse k-neighbour matvec -- a dense (n_units, n_units)
                    # matmul every 40us frame is not tractable at n_units in
                    # the thousands.
                    drive = drive + W.dot(fired_prev)
                else:
                    # Element-wise multiply + row-sum rather than `W @ fired_prev`:
                    # numpy's BLAS matmul raises spurious FP-flag warnings on this
                    # tiny mostly-zero product even when operands are finite. Same
                    # math, no BLAS.
                    drive = drive + (W * fired_prev).sum(axis=1)

            in_refractory = ts < refr_until
            V += leak_coeff * (v_rest - V) + drive
            # Absolute refractory: pin V to reset rather than let it keep
            # integrating input. Without this, a neuron receiving sustained
            # coupling drive while refractory climbs unbounded -> float overflow.
            V[in_refractory] = self._v_reset
            # Finite-range guard. The bounds sit far outside any physiological
            # membrane voltage, so this never distorts real dynamics; it only
            # keeps a pathological coupling "seizure" finite instead of NaN.
            np.clip(V, -150.0, 100.0, out=V)

            # --- threshold / reset -----------------------------------------
            # spiking_mask gates ELECTRODES in 1:1 mode (grounded/reference
            # channels never fire). In spatial mode every neuron may fire;
            # electrode gating instead happens in the projection (NON_SPIKING
            # channels are already zeroed there), so no mask is applied here.
            if spatial:
                fired = (V >= self._v_thresh) & ~in_refractory
            else:
                fired = (V >= self._v_thresh) & ~in_refractory & self._spiking_mask
            V[fired] = self._v_reset
            refr_until[fired] = ts + self._refractory_frames

            # --- eligibility: co-activity that a later reward will consolidate ---
            if plastic:
                fired_now = fired.astype(np.float64)
                elig *= elig_decay
                if stdp:
                    # Trace-based STDP: a firing neuron credits synapses from
                    # partners active over the last ~tau (a real ms-scale window),
                    # not just the previous frame. Traces are bumped AFTER pairing
                    # so a spike never pairs with itself.
                    x_pre  *= pre_decay
                    y_post *= post_decay
                    if fired_now.any():
                        elig += np.outer(fired_now, x_pre)            # causal LTP (pre-before-post)
                        elig -= ltd_ratio * np.outer(y_post, fired_now)  # acausal LTD (post-before-pre)
                    x_pre  += fired_now
                    y_post += fired_now
                else:
                    # Original 1-frame (40 us) pre->post coincidence rule.
                    elig += np.outer(fired_now, fired_prev)
                fired_prev = fired_now
            else:
                fired_prev = fired.astype(np.float64)

            # --- emit spikes + stamp extracellular waveform into the trace --
            if fired.any():
                if spatial:
                    for u in np.nonzero(fired)[0]:
                        u = int(u)
                        home = int(self._home_channel[u])
                        # samples reported for a spike are always relative to
                        # the electrode it's attributed to (its home channel,
                        # i.e. the electrode this neuron projects to loudest).
                        spikes.append(DataSourceSpike(
                            timestamp           = ts,
                            channel             = home,
                            samples             = (self._spike_template * self._projection[u, home]).astype(np.float32),
                            channel_mean_sample = 0.0,
                        ))
                        self._stamp_waveform_spatial(ts, u)
                else:
                    for ch in np.nonzero(fired)[0]:
                        ch = int(ch)
                        spikes.append(DataSourceSpike(
                            timestamp           = ts,
                            channel             = ch,
                            samples             = self._spike_template.copy(),
                            channel_mean_sample = 0.0,
                        ))
                        self._stamp_waveform(ts, ch)

            # --- pull this frame's already-scheduled waveform contribution --
            wf = self._pending_waveform.pop(ts, None)
            if wf is not None:
                frames_uV[step] += wf

            # --- optional cosmetic stim artifact ---------------------------
            if self._stim_artifact and I_ext is not None:
                frames_uV[step] += self._artifact_gain * I_ext

        self._V = V
        self._fired_prev = fired_prev
        self._next_read_ts = to_timestamp

        frames = np.clip(
            np.round(frames_uV / UV_PER_SAMPLE_UNIT), INT16_MIN, INT16_MAX
        ).astype(np.int16)
        return DataSourceBatch(frames=np.ascontiguousarray(frames), spikes=spikes)

    # -- plasticity (opt-in) -------------------------------------------------

    def apply_reward(self, reward: float) -> None:
        """Consolidate (reward > 0) or depress (reward < 0) recently co-active
        coupling pairs. No-op unless ``plasticity=True``. Call between closed-loop
        episodes.

        This is the third factor of a three-factor rule: the eligibility trace holds
        *what fired together*, and the global reward decides whether to strengthen
        it. The trace is reset afterwards so each reward consolidates only the
        activity accumulated since the previous one.
        """
        if not self._plastic or self._elig is None or self._W is None:
            return
        self._W += self._plasticity_lr * float(reward) * self._elig
        np.clip(self._W, 0.0, self._weight_max, out=self._W)   # excitatory, bounded
        np.fill_diagonal(self._W, 0.0)                          # no self-coupling
        self._W[list(NON_SPIKING), :] = 0.0
        self._W[:, list(NON_SPIKING)] = 0.0
        self._elig[...] = 0.0

    @property
    def coupling_weights(self) -> np.ndarray:
        """Current coupling weight matrix (mV), shape (64, 64). Returns a copy."""
        assert self._W is not None, "coupling_weights before open()"
        return self._W.copy()

    @property
    def connectivity_matrix(self) -> np.ndarray:
        """The structural coupling graph (64, 64), before scaling by coupling_gain.
        Returns a copy."""
        assert self._adjacency is not None, "connectivity_matrix before open()"
        return self._adjacency.copy()

    def set_coupling_gain(self, gain_mV: float) -> None:
        """Reset coupling to the nearest-neighbour structure at a new gain. Useful
        for live control (e.g. an interactive visualiser). Overwrites any learned
        weights, so do not use mid-experiment with plasticity on."""
        self._coupling_gain = float(gain_mV)
        self._use_coupling = (self._coupling_gain > 0.0) or self._plastic
        if self._adjacency is not None:
            self._W = self._coupling_gain * self._adjacency.copy()

    # -- helpers -------------------------------------------------------------

    def _stamp_waveform(self, ts: int, channel: int) -> None:
        """Add the canonical spike waveform to frames [ts, ts+75), possibly spanning
        into the next read() batch (hence a persistent dict)."""
        template = self._spike_template
        for k in range(SPIKE_SAMPLES_TOTAL):
            fts = ts + k
            vec = self._pending_waveform.get(fts)
            if vec is None:
                vec = np.zeros(CHANNEL_COUNT, dtype=np.float64)
                self._pending_waveform[fts] = vec
            vec[channel] += template[k]

    def _stamp_waveform_spatial(self, ts: int, unit: int) -> None:
        """Spatial-mode version of _stamp_waveform: one firing neuron is seen
        by every electrode within its projection, not just one channel, each
        scaled by that electrode's share of the neuron's amplitude."""
        template = self._spike_template          # unit-amplitude, trough -1.0
        channels, amps = self._unit_channels[unit]
        for k in range(SPIKE_SAMPLES_TOTAL):
            fts = ts + k
            vec = self._pending_waveform.get(fts)
            if vec is None:
                vec = np.zeros(CHANNEL_COUNT, dtype=np.float64)
                self._pending_waveform[fts] = vec
            vec[channels] += template[k] * amps

    def _drop_stale_pending(self, from_timestamp: int) -> None:
        """Guard against unbounded growth: the producer reads timestamps
        monotonically, so anything keyed before the current read head can never be
        consumed. In normal operation this removes nothing."""
        for store in (self._pending_current, self._pending_waveform):
            stale = [ts for ts in store if ts < from_timestamp]
            for ts in stale:
                del store[ts]

    def _resolve_connectivity(self) -> np.ndarray:
        """Return the structural coupling matrix: the user-supplied graph if one
        was passed, otherwise a freshly generated weighted small-world graph."""
        if self._connectivity_arg is not None:
            W = np.array(self._connectivity_arg, dtype=np.float64)
            if W.shape != (CHANNEL_COUNT, CHANNEL_COUNT):
                raise ValueError(
                    f"connectivity must have shape ({CHANNEL_COUNT}, {CHANNEL_COUNT}), got {W.shape}"
                )
            W[list(NON_SPIKING), :] = 0.0   # never drive from/into dead channels
            W[:, list(NON_SPIKING)] = 0.0
            np.fill_diagonal(W, 0.0)
            return W
        return small_world_connectivity(
            k            = self._sw_k,
            rewire_p     = self._sw_rewire_p,
            weight_sigma = self._sw_weight_sigma,
            seed         = self._connectivity_seed,
        )

    def _build_spike_template(self) -> np.ndarray:
        """Canonical biphasic extracellular action potential: a sharp negative
        trough at the detection sample, then a slower positive rebound. Mean-centred
        (per the SDK convention) and scaled to the configured amplitude.

        In spatial mode the template is left at unit amplitude (trough -1.0);
        the per-electrode scale lives in self._projection instead, since one
        neuron's spike has a different amplitude on every electrode that sees
        it. In 1:1 mode there is exactly one electrode per neuron, so the
        scale is baked in here as before."""
        idx = np.arange(SPIKE_SAMPLES_TOTAL, dtype=np.float64)
        trough  = -np.exp(-((idx - SPIKE_SAMPLES_BEFORE) / 4.0) ** 2)
        rebound = 0.4 * np.exp(-((idx - (SPIKE_SAMPLES_BEFORE + 15)) / 8.0) ** 2)
        template = trough + rebound
        template -= template.mean()
        template /= -template.min()          # trough -> -1.0
        if not self._spatial:
            template *= self._spike_amplitude_uV  # trough -> -spike_amplitude_uV
        return np.ascontiguousarray(template, dtype=np.float32)

    def _setup_spatial(self) -> None:
        """Place n_units neurons over the 8x8 electrode footprint and build
        the electrode <-> neuron projection used by read() and on_stims()."""
        elec_pos = _electrode_positions()
        self._unit_pos = _generate_unit_positions(self._n_units, self._pos_seed, self._pos_pad)
        # A separate stream from placement so amplitude and position don't
        # covary through a shared draw order.
        amp_rng = np.random.default_rng(self._pos_seed + 1)
        peak_uv = amp_rng.lognormal(
            mean=np.log(self._spike_amplitude_uV), sigma=self._amp_lognorm_sig,
            size=self._n_units,
        )
        d = np.sqrt(((self._unit_pos[:, None, :] - elec_pos[None, :, :]) ** 2).sum(axis=2))
        self._projection = peak_uv[:, None] * np.exp(-d / self._proj_sigma)
        self._projection[d > self._proj_cutoff * self._proj_sigma] = 0.0
        self._projection[:, list(NON_SPIKING)] = 0.0
        # unit_position_pad keeps every neuron within self._proj_cutoff of its
        # nearest electrode, so no row of the projection is ever all-zero.
        self._home_channel = self._projection.argmax(axis=1)
        self._unit_channels = [
            (np.nonzero(row)[0], row[np.nonzero(row)[0]]) for row in self._projection
        ]
        # Spatial kernel (0..1, not amplitude-scaled) for spreading a channel's
        # stim current to nearby neurons -- the stimulus's field, not a spike's.
        spread = np.exp(-d / self._proj_sigma)
        spread[d > self._proj_cutoff * self._proj_sigma] = 0.0
        spread[:, list(NON_SPIKING)] = 0.0
        self._stim_spread = spread


def make_lif_source(**kwargs) -> LIFDataSource:
    """Factory for ``cl.sim.set_simulator_data_source``. All kwargs are forwarded to
    :class:`LIFDataSource` and must be JSON-serialisable."""
    return LIFDataSource(**kwargs)
