"""Attention scores (Q·Kᵀ) computed by coincidence detection in the stock LIF simulator.

The question this answers: given a binary query q and four binary keys k_0..k_3,
which key matches q best? In a transformer that is argmax_i q·k_i. Here the
silicon only ENCODES the vectors as pulse timing and DECODES spike counts; the
products q_j·k_ij come from the neurons' membranes.

Layout: slot i (one key) uses 8 electrodes, channels 8+8i .. 15+8i, one per
dimension j. (The viewer draws channel c at row c%8, column c//8, so each slot
is one COLUMN of the 8x8 grid: columns 1-4.)

Encoding, per trial:
  t_k       pulse electrode (i,j) if k_ij = 1      (the key "primes" the membrane)
  t_k + Δ   pulse electrode (i,j) if q_j  = 1      (the query, broadcast to every slot)

Each pulse alone lifts the membrane by 20 mV/µA x amplitude (4 frames x 5 mV/µA),
below the 20 mV gap to threshold. The first pulse decays with tau = 20 ms, so a
second pulse Δ later lands on top of exp(-Δ/20 ms) of the first. With the right
amplitude, "both pulses" crosses threshold far more often than "one pulse": an AND
gate, i.e. multiplication of two bits. Slot i's spike count in the window right
after the query pulse then estimates q·k_i, and argmax over slots picks the key.

Δ cannot go below 5 ms: the SDK allows at most one pulse per electrode per 5 ms
(its 200 Hz per-channel limit), so key and query pulses on the same electrode are
at least 5 ms apart.

Simulator configuration: the stock LIF source with every default untouched, except
coupling_gain_mV (0 = independent neurons, the .env default; 4.0 = example1.py).

Run from the repo root:

    CL_SDK_ACCELERATED_TIME=1 uv run python experiments/attention_reservoir.py --calibrate --no-hold
    CL_SDK_ACCELERATED_TIME=1 uv run python experiments/attention_reservoir.py --trials 600 --no-hold
    CL_SDK_ACCELERATED_TIME=1 uv run python experiments/attention_reservoir.py --coupling 4.0 --no-hold
    uv run python experiments/attention_reservoir.py --realtime-demo --trials 20
"""
from __future__ import annotations

import argparse
import os
import webbrowser
from dataclasses import dataclass

import numpy as np

import cl

# ----------------------------------------------------------------------------- layout
SLOTS, DIMS = 4, 8
FIRST_CHANNEL = 8                       # slot i, dim j -> channel 8 + 8i + j  (channels 8..39)
KEY_BITS = 4                            # balanced keys: every key has exactly 4 ones

FRAME_US = 40                           # 25 kHz
FRAMES_PER_MS = 25
TICKS_PER_SECOND = 10                   # one tick = 100 ms; membranes relax fully between trials

PULSE_PHASE_US = 160                    # 4 frames of cathodic current, then the charge-balancing phase
BASE_LEAD_US = 20_000                   # schedule 20 ms ahead so real-time mode never delivers late
CHANNEL_BUSY_FRAMES = 123               # SDK: after a single pulse a channel is busy for 5 ms - 80 µs
MIN_DELTA_MS = 5.0                      # => key->query gap on one electrode is at least 5 ms
WINDOW_MS = 2.0                         # readout window after each pulse time


def channel(slot: int, dim: int) -> int:
    return FIRST_CHANNEL + slot * DIMS + dim


ALL_CHANNELS = np.array([[channel(i, j) for j in range(DIMS)] for i in range(SLOTS)])   # (4, 8)


# ----------------------------------------------------------------------------- stimulation
@dataclass
class Presentation:
    """One key-then-query pulse pattern, and when it was delivered."""
    key_mask: np.ndarray     # (4, 8) bool: pulse at t_k
    query_mask: np.ndarray   # (4, 8) bool: pulse at t_k + Δ
    key_amp_uA: float
    query_amp_uA: float
    delta_frames: int
    t_key: int = -1          # timestamps (25 kHz frames), filled in when scheduled
    t_query: int = -1


def schedule(neurons, p: Presentation, tick_ts: int) -> None:
    """Queue the key pulses at tick_ts + 20 ms and the query pulses Δ later.

    The SDK times a pulse on a busy channel from when the channel frees up, not from
    the tick. Electrodes that already got a key pulse this trial are busy for 123
    frames, so their query pulse needs a shorter lead. Both groups land on the same
    frame; the run checks this against the recorded stim timestamps.
    """
    key_design = cl.StimDesign(PULSE_PHASE_US, -p.key_amp_uA, PULSE_PHASE_US, p.key_amp_uA)
    query_design = cl.StimDesign(PULSE_PHASE_US, -p.query_amp_uA, PULSE_PHASE_US, p.query_amp_uA)
    key_chs = ALL_CHANNELS[p.key_mask].tolist()
    q_after_key = ALL_CHANNELS[p.query_mask & p.key_mask].tolist()
    q_only = ALL_CHANNELS[p.query_mask & ~p.key_mask].tolist()

    p.t_key = tick_ts + BASE_LEAD_US // FRAME_US
    p.t_query = p.t_key + p.delta_frames
    if key_chs:
        neurons.stim(cl.ChannelSet(*key_chs), key_design, lead_time_us=BASE_LEAD_US)
    if q_after_key:
        neurons.stim(cl.ChannelSet(*q_after_key), query_design,
                     lead_time_us=(p.delta_frames - CHANNEL_BUSY_FRAMES) * FRAME_US)
    if q_only:
        neurons.stim(cl.ChannelSet(*q_only), query_design,
                     lead_time_us=BASE_LEAD_US + p.delta_frames * FRAME_US)


def configure_simulator(seed: int, coupling: float) -> None:
    if cl.is_simulator():
        cl.sim.set_simulator_data_source(
            "organoid_simulator.lif_data_source:make_lif_source",
            config={"random_seed": seed, "coupling_gain_mV": coupling},   # everything else: stock defaults
        )


def run_presentations(presentations: list[Presentation], *, seed: int, coupling: float,
                      ticks_between: int = 1, on_tick=None, open_browser: bool = False,
                      hold: bool = False):
    """Deliver every presentation (one per `ticks_between` ticks) and return all spikes and stims."""
    configure_simulator(seed, coupling)
    spike_ts, spike_ch, stim_ts, stim_ch = [], [], [], []
    n_ticks = len(presentations) * ticks_between + 2

    with cl.open() as neurons:
        server = getattr(neurons, "_websocket_server", None)
        viewer_url = server.web_url if server is not None else None
        if viewer_url and open_browser:
            webbrowser.open(viewer_url)

        next_p = 0
        for tick in neurons.loop(ticks_per_second=TICKS_PER_SECOND, stop_after_ticks=n_ticks):
            for s in tick.analysis.spikes:
                spike_ts.append(s.timestamp)
                spike_ch.append(s.channel)
            for s in tick.analysis.stims:
                stim_ts.append(s.timestamp)
                stim_ch.append(s.channel)
            if on_tick is not None:
                on_tick(tick.iteration, spike_ts, spike_ch, presentations[:next_p])
            if next_p < len(presentations) and tick.iteration % ticks_between == 0:
                schedule(neurons, presentations[next_p], tick.iteration_timestamp)
                next_p += 1

        if viewer_url and hold:
            print(f"\nViewer still running at {viewer_url}")
            input("Press Enter to close it and exit ...")

    order = np.argsort(spike_ts, kind="stable")
    spikes = (np.asarray(spike_ts)[order], np.asarray(spike_ch, dtype=int)[order])
    stims = (np.asarray(stim_ts), np.asarray(stim_ch, dtype=int))
    return spikes, stims


def counts_in_window(spikes, start: int, stop: int) -> np.ndarray:
    """Spike count per channel (64,) with timestamp in [start, stop)."""
    ts, ch = spikes
    a, b = np.searchsorted(ts, [start, stop])
    return np.bincount(ch[a:b], minlength=64)


def check_timing(presentations: list[Presentation], stims) -> None:
    """Every recorded pulse must sit exactly at its trial's t_key or t_query."""
    stim_ts, stim_ch = stims
    expected = set()
    for p in presentations:
        expected |= {(p.t_key, c) for c in ALL_CHANNELS[p.key_mask].tolist()}
        expected |= {(p.t_query, c) for c in ALL_CHANNELS[p.query_mask].tolist()}
    got = set(zip(stim_ts.tolist(), stim_ch.tolist()))
    if got != expected:
        raise RuntimeError(f"stim timing mismatch: {len(expected - got)} missing, {len(got - expected)} unexpected")


# ----------------------------------------------------------------------------- calibration
def calibrate(args) -> None:
    """Sweep (key amplitude, query amplitude, Δ) and measure P(fire) per pulse condition."""
    sweep = [(a, a, d) for a in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1) for d in (5.0, 7.0, 10.0)]   # symmetric
    sweep += [(ka, qa, 5.0) for ka in (0.4, 0.5, 0.6, 0.7) for qa in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
              if ka != qa]                                                                      # asymmetric
    reps = args.cal_reps
    rng = np.random.default_rng(args.seed)

    # Every electrode in every presentation gets a random condition: no pulse, key only,
    # query only, or both. 32 electrodes x reps presentations per cell.
    presentations, cells = [], []
    for cell in sweep:
        ka, qa, d = cell
        for _ in range(reps):
            cond = rng.integers(0, 4, size=(SLOTS, DIMS))
            presentations.append(Presentation(cond & 1 == 1, cond & 2 == 2, ka, qa, round(d * FRAMES_PER_MS)))
            cells.append(cell)
    order = rng.permutation(len(presentations))
    presentations = [presentations[i] for i in order]
    cells = [cells[i] for i in order]

    print(f"Calibrating: {len(presentations)} presentations, coupling_gain_mV={args.coupling} ...")
    spikes, stims = run_presentations(presentations, seed=args.seed, coupling=args.coupling)
    check_timing(presentations, stims)
    W = round(WINDOW_MS * FRAMES_PER_MS)

    # Latency check: evoked spikes after a query pulse, relative to the pulse.
    lat = []
    ts, ch = spikes
    for p in presentations:
        a, b = np.searchsorted(ts, [p.t_query, p.t_query + 5 * FRAMES_PER_MS])
        pulsed = set(ALL_CHANNELS[p.query_mask].tolist())
        lat += [(t - p.t_query) for t, c in zip(ts[a:b], ch[a:b]) if c in pulsed]
    lat = np.asarray(lat)
    if lat.size:
        print(f"  spike latency after query pulse: median {np.median(lat) / FRAMES_PER_MS:.2f} ms, "
              f"{np.mean(lat < W) * 100:.0f}% of spikes within 5 ms fall in the {WINDOW_MS:g} ms window")

    # P(fire) per condition. 'any' = spike anywhere from t_k to t_q + window;
    # 'q-win' = spike in the window right after t_q (what readout A counts).
    stats: dict = {}
    for p, cell in zip(presentations, cells):
        any_win = counts_in_window(spikes, p.t_key, p.t_query + W)[ALL_CHANNELS] > 0
        q_win = counts_in_window(spikes, p.t_query, p.t_query + W)[ALL_CHANNELS] > 0
        cond = p.key_mask.astype(int) + 2 * p.query_mask.astype(int)
        for c in range(4):
            s = stats.setdefault(cell, {}).setdefault(c, [0, 0, 0])
            m = cond == c
            s[0] += int(m.sum()); s[1] += int(any_win[m].sum()); s[2] += int(q_win[m].sum())

    print("\nCalibration: P(neuron fires) by pulses received.  ΔV per pulse = 20 mV/µA; gap to threshold = 20 mV.")
    print("  any   = spike anywhere in t_k .. t_q+2ms        q-win = spike in t_q .. t_q+2ms (readout A)")
    print("  inter = P11 - P10 - P01 + P00 (q-win): the only part of a slot's count that depends on q·k")
    print("  d'    = inter / std of the difference of two slot counts (|q| = 4, q·k = 2): one presentation,")
    print("          one unit of q·k margin. d' ~ 1 needs (1/d')^2 repeated presentations.")
    print(f"  {'key µA':>6} {'qry µA':>6} {'Δ ms':>5} | {'any: none':>9} {'key':>5} {'query':>5} {'both':>5} | "
          f"{'q-win: none':>11} {'key':>5} {'query':>5} {'both':>5} | {'inter':>6} {'d_prime':>7}")
    for cell in sweep:
        s = stats[cell]
        pa = [s[c][1] / max(s[c][0], 1) for c in range(4)]
        pq = [s[c][2] / max(s[c][0], 1) for c in range(4)]
        inter = pq[3] - pq[1] - pq[2] + pq[0]
        slot_var = 2 * sum(p * (1 - p) for p in pq)          # 2 electrodes in each of the 4 conditions
        dprime = inter / np.sqrt(2 * slot_var) if slot_var > 0 else 0.0
        ka, qa, d = cell
        print(f"  {ka:6.1f} {qa:6.1f} {d:5.1f} | {pa[0]:9.2f} {pa[1]:5.2f} {pa[2]:5.2f} {pa[3]:5.2f} | "
              f"{pq[0]:11.2f} {pq[1]:5.2f} {pq[2]:5.2f} {pq[3]:5.2f} | {inter:6.2f} {dprime:7.2f}")


# ----------------------------------------------------------------------------- task
def make_trials(n: int, rng: np.random.Generator):
    """Random binary queries and 4 balanced keys (4 ones each), with a unique best key."""
    queries, keys = [], []
    while len(queries) < n:
        q = rng.integers(0, 2, size=DIMS)
        if q.sum() == 0:
            continue
        k = np.zeros((SLOTS, DIMS), dtype=int)
        for i in range(SLOTS):
            k[i, rng.choice(DIMS, KEY_BITS, replace=False)] = 1
        scores = k @ q
        if (scores == scores.max()).sum() != 1:
            continue                      # skip ties: the target must be unambiguous
        queries.append(q); keys.append(k)
    return np.array(queries), np.array(keys)


def argmax_random_ties(x: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(x == x.max())
    return int(rng.choice(best))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fit_softmax_regression(X_tr, y_tr, X_te, l2: float = 1e-2, steps: int = 2000, lr: float = 0.5):
    """Multinomial logistic regression (numpy, full-batch gradient descent). Returns test predictions."""
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    A, B = (X_tr - mu) / sd, (X_te - mu) / sd
    A = np.hstack([A, np.ones((len(A), 1))]); B = np.hstack([B, np.ones((len(B), 1))])
    Y = np.eye(SLOTS)[y_tr]
    Wt = np.zeros((A.shape[1], SLOTS))
    for _ in range(steps):
        P = softmax(A @ Wt)
        grad = A.T @ (P - Y) / len(A) + l2 * Wt
        Wt -= lr * grad
    return (B @ Wt).argmax(1)


def evaluate(args) -> None:
    rng = np.random.default_rng(args.seed)
    if args.delta_ms < MIN_DELTA_MS:
        raise SystemExit(f"--delta-ms must be >= {MIN_DELTA_MS}: the SDK allows one pulse per electrode per 5 ms")
    delta_frames = round(args.delta_ms * FRAMES_PER_MS)
    queries, keys = make_trials(args.trials, rng)
    target = (keys @ queries[:, :, None])[:, :, 0].argmax(1)

    presentations = []
    for q, k in zip(queries, keys):
        for _ in range(args.repeats):
            presentations.append(Presentation(k == 1, np.broadcast_to(q == 1, (SLOTS, DIMS)).copy(),
                                              args.key_amp_ua, args.amp_ua, delta_frames))

    W = round(WINDOW_MS * FRAMES_PER_MS)
    on_tick = None
    ticks_between = 1
    if args.realtime_demo:
        ticks_between = 2                                     # 200 ms between presentations
        on_tick = make_demo_printer(queries, keys, args.repeats, ticks_between, W)

    print(f"Running {args.trials} trials x {args.repeats} presentation(s): key {args.key_amp_ua} µA / query {args.amp_ua} µA "
          f"({20 * args.key_amp_ua:.0f} / {20 * args.amp_ua:.0f} mV per pulse), Δ={args.delta_ms} ms, coupling_gain_mV={args.coupling}, seed={args.seed}")
    spikes, stims = run_presentations(presentations, seed=args.seed, coupling=args.coupling,
                                      ticks_between=ticks_between, on_tick=on_tick,
                                      open_browser=args.realtime_demo and not args.no_browser,
                                      hold=args.realtime_demo and not args.no_hold)
    check_timing(presentations, stims)

    # Per-trial features, summed over repeats.
    q_counts = np.zeros((args.trials, 64)); k_counts = np.zeros((args.trials, 64))
    for n, p in enumerate(presentations):
        t = n // args.repeats
        q_counts[t] += counts_in_window(spikes, p.t_query, p.t_query + W)
        k_counts[t] += counts_in_window(spikes, p.t_key, p.t_key + W)
    slot_spikes = q_counts[:, ALL_CHANNELS].sum(2)                                  # (trials, 4)
    slot_spikes_all = (q_counts + k_counts)[:, ALL_CHANNELS].sum(2)
    slot_pulses = keys.sum(2) + queries.sum(1, keepdims=True)                      # (trials, 4), all equal
    true_scores = (keys @ queries[:, :, None])[:, :, 0]

    tie_rng = np.random.default_rng(args.seed + 1)
    acc = lambda scores: np.mean([argmax_random_ties(s, tie_rng) == y for s, y in zip(scores, target)])

    # Readout B: linear softmax regression, train on the first half, test on the second.
    half = args.trials // 2
    X_neurons = np.hstack([q_counts, k_counts])                                     # 128 spike features
    X_stim = np.hstack([keys.reshape(len(keys), -1), np.repeat(queries, SLOTS, axis=1)])  # 64 pulse indicators
    accB_neurons = np.mean(fit_softmax_regression(X_neurons[:half], target[:half], X_neurons[half:]) == target[half:])
    accB_stim = np.mean(fit_softmax_regression(X_stim[:half], target[:half], X_stim[half:]) == target[half:])

    print("\nResults  (chance = 25%; keys balanced at 4 ones; ties in q·k excluded)")
    print(f"  {'readout':<62} {'accuracy':>8}")
    print(f"  {'A  neurons: argmax of slot spike count, window after query':<62} {acc(slot_spikes):8.1%}")
    print(f"  {'A' + chr(39) + ' neurons: same, whole trial window (key + query)':<62} {acc(slot_spikes_all):8.1%}")
    print(f"  {'control A: argmax of pulses per slot (silicon, no neurons)':<62} {acc(slot_pulses):8.1%}")
    print(f"  {'perfect silicon: exact q·k in Python':<62} {acc(true_scores):8.1%}")
    print(f"  {f'B  neurons: softmax regression on 128 spike counts (test n={args.trials - half})':<62} {accB_neurons:8.1%}")
    print(f"  {'control B: same regression on 64 raw pulse indicators':<62} {accB_stim:8.1%}")

    # Within-trial view: subtract each trial's mean so the shared |q| effect cancels.
    ds, dt = slot_spikes - slot_spikes.mean(1, keepdims=True), true_scores - true_scores.mean(1, keepdims=True)
    print(f"\n  within-trial correlation(slot spike count, q·k) = {np.corrcoef(ds.ravel(), dt.ravel())[0, 1]:.2f}")
    print(f"  spikes per unit of q·k within a trial = {(ds * dt).sum() / (dt * dt).sum():.2f} "
          f"(per presentation: {(ds * dt).sum() / (dt * dt).sum() / args.repeats:.2f})")

    print("\nAttention weights, first 5 trials: softmax(spikes per presentation) vs softmax(true q·k)")
    for t in range(min(5, args.trials)):
        print(f"  q={''.join(map(str, queries[t]))}  q·k={true_scores[t].tolist()}  spikes={slot_spikes[t].astype(int).tolist()}")
        print(f"      neurons {np.array2string(softmax(slot_spikes[t] / args.repeats), precision=2)}   "
              f"true {np.array2string(softmax(true_scores[t].astype(float)), precision=2)}")


def make_demo_printer(queries, keys, repeats, ticks_between, W):
    """Print each trial's slot spike counts as soon as its last presentation is over."""
    done = {"t": 0}

    def on_tick(iteration, spike_ts, spike_ch, scheduled):
        t = done["t"]
        last = (t + 1) * repeats - 1
        if t >= len(queries) or last >= len(scheduled):
            return
        p_last = scheduled[last]
        if not spike_ts or spike_ts[-1] < p_last.t_query + W:
            return
        ts, ch = np.asarray(spike_ts), np.asarray(spike_ch)
        counts = np.zeros(64)
        for p in scheduled[t * repeats:last + 1]:
            m = (ts >= p.t_query) & (ts < p.t_query + W)
            counts += np.bincount(ch[m], minlength=64)
        slot = counts[ALL_CHANNELS].sum(1)
        true = keys[t] @ queries[t]
        mark = "OK " if slot.argmax() == true.argmax() else "-- "
        print(f"  trial {t:3d} {mark} q={''.join(map(str, queries[t]))}  q·k={true.tolist()}  "
              f"spikes/slot={slot.astype(int).tolist()}  (slot i = viewer column i+1)")
        done["t"] += 1

    return on_tick


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=600, help="Number of query/keys trials (default 600)")
    parser.add_argument("--seed", type=int, default=42, help="Simulator and dataset seed (default 42)")
    parser.add_argument("--coupling", type=float, default=0.0, choices=[0.0, 4.0],
                        help="LIF coupling_gain_mV: 0 (.env default) or 4.0 (example1.py) (default 0)")
    parser.add_argument("--delta-ms", type=float, default=5.0, help="Key->query pulse gap, >= 5 ms (default 5)")
    parser.add_argument("--amp-ua", type=float, default=0.5, help="Query pulse amplitude in µA, <= 3 (default 0.5)")
    parser.add_argument("--key-amp-ua", type=float, default=None,
                        help="Key pulse amplitude in µA, <= 3 (default: same as --amp-ua)")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Presentations per trial, spikes summed (default 1; 5 with --realtime-demo)")
    parser.add_argument("--calibrate", action="store_true", help="Sweep amplitude and Δ, print P(fire) table")
    parser.add_argument("--cal-reps", type=int, default=40, help="Presentations per calibration cell (default 60)")
    parser.add_argument("--realtime-demo", action="store_true",
                        help="Slow pacing for the SDK viewer: 5 presentations 200 ms apart per trial")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the viewer in a browser")
    parser.add_argument("--no-hold", action="store_true", help="Exit as soon as the experiment ends")
    args = parser.parse_args()
    if args.key_amp_ua is None:
        args.key_amp_ua = args.amp_ua
    if not (0 < args.amp_ua <= 3.0 and 0 < args.key_amp_ua <= 3.0):
        parser.error("amplitudes must be in (0, 3] µA")
    if args.repeats is None:
        args.repeats = 5 if args.realtime_demo else 1
    if args.realtime_demo and os.environ.get("CL_SDK_ACCELERATED_TIME") == "1":
        print("Note: CL_SDK_ACCELERATED_TIME=1 disables the viewer; unset it for the demo.")
    return args


def main() -> None:
    args = parse_args()
    if args.calibrate:
        calibrate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
