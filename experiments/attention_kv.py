"""Attention block (key -> value memory) learned by simulated neurons via STDP.

The silicon side does no learning. It only:
  * encodes: chooses which electrodes to stimulate and when, and
  * decodes: counts spikes per electrode group after a query.
The K->V association is written into the culture's coupling weights by
spike-timing-dependent plasticity, purely from stimulation timing.

Layout (channel = row*8 + col):

    row 1-2  keys    K0={8,9,16,17}  K1={10,11,18,19}  K2={12,13,20,21}  K3={14,15,22,23}
    row 3-4  (gap, unused)
    row 5-6  values  V0={40,41,48,49} V1={42,43,50,51}  V2={44,45,52,53}  V3={46,47,54,55}

Phases (one cl.open() session, so learned weights persist):
  1. baseline  query each Ki alone; count spikes in each Vj group (expect ~0)
  2. train     stim Ki, then Vi `--pair-gap-ms` later (pre-before-post -> LTP on Ki->Vi)
  3. recall    query each Ki alone; R[i][j] = mean spikes in Vj
  4. blend     query a mix of K1/K2 electrodes (4/0 .. 0/4); V1 vs V2 follows the mix
  --reverse    trains Vi before Ki (post-before-pre -> LTD); recall should fail

Timing: both stims of a pair are issued in the same loop tick. Inside a loop the
SDK anchors every stim command to the tick timestamp, and `lead_time_us` offsets
the pulse from it in 40 us steps, so `lead_time_us=80` vs `80 + gap*1000` gives an
exact gap independent of the tick rate.

Run from the repo root:

    CL_SDK_ACCELERATED_TIME=1 uv run python experiments/attention_kv.py --no-hold
    CL_SDK_ACCELERATED_TIME=1 uv run python experiments/attention_kv.py --no-hold --reverse
    uv run python experiments/attention_kv.py --realtime-demo        # watch it in the viewer
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import webbrowser
from collections import defaultdict

import numpy as np

import cl

KEYS = [
    [8, 9, 16, 17],
    [10, 11, 18, 19],
    [12, 13, 20, 21],
    [14, 15, 22, 23],
]
VALUES = [
    [40, 41, 48, 49],
    [42, 43, 50, 51],
    [44, 45, 52, 53],
    [46, 47, 54, 55],
]
N = len(KEYS)
STIM = cl.StimDesign(200, -1.5, 200, 1.5)   # biphasic, cathodic-first; reliably fires the stimulated cell
FRAMES_PER_MS = 25                          # 25 kHz external timestamps
BASE_LEAD_US = 80                           # SDK minimum lead time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-reps", type=int, default=8, help="Queries per key in baseline (default 8)")
    p.add_argument("--train-reps", type=int, default=40, help="Training pairings per K/V pair (default 40)")
    p.add_argument("--recall-reps", type=int, default=10, help="Queries per key in recall (default 10)")
    p.add_argument("--blend-reps", type=int, default=10, help="Queries per mix in the blend sweep (default 10)")
    p.add_argument("--pair-gap-ms", type=float, default=5.0, help="K->V gap during training (default 5 ms)")
    p.add_argument("--spacing-ms", type=float, default=150.0,
                   help="Time between events; >> STDP window so pairs never overlap (default 150)")
    p.add_argument("--window-ms", type=float, default=30.0, help="Post-query readout window (default 30 ms)")
    p.add_argument("--weight-max", type=float, default=8.0, help="LIF weight_max_mV cap on learned weights (default 8)")
    p.add_argument("--lr", type=float, default=0.15, help="LIF plasticity_lr (mV per unit STDP eligibility, default 0.15)")
    p.add_argument("--ltd-ratio", type=float, default=1.2,
                   help="LIF stdp_ltd_ratio; >1 so uncorrelated background firing nets to depression (default 1.2)")
    p.add_argument("--bg-drive", type=float, default=0.4,
                   help="LIF background_drive_std_mV; lower = quieter culture, cleaner readout (default 0.4)")
    p.add_argument("--sim-config", type=str, default="{}", help="Extra JSON merged into the LIF config")
    p.add_argument("--reverse", action="store_true", help="Control: train Vi BEFORE Ki (should not learn)")
    p.add_argument("--seed", type=int, default=42, help="Simulator random seed (default 42)")
    p.add_argument("--no-browser", action="store_true", help="Do not open the viewer in a browser")
    p.add_argument("--no-hold", action="store_true", help="Exit as soon as the experiment ends")
    p.add_argument("--realtime-demo", action="store_true",
                   help="Pacing for watching in the viewer: 300 ms between events, fewer test reps (~65 s)")
    args = p.parse_args()
    if args.realtime_demo:
        args.spacing_ms = max(args.spacing_ms, 300.0)
        args.baseline_reps, args.recall_reps, args.blend_reps = 3, 4, 4
    return args


def build_schedule(args) -> list[tuple[str, object]]:
    """Ordered list of (phase, payload) events, one per `spacing_ms` slot."""
    rng = np.random.default_rng(args.seed)
    ev: list[tuple[str, object]] = []
    for _ in range(args.baseline_reps):
        for i in rng.permutation(N):
            ev.append(("baseline", int(i)))
        ev.append(("baseline", None))                     # no-stim slot: spontaneous-rate reference
    for _ in range(args.train_reps):
        for i in rng.permutation(N):
            ev.append(("train", int(i)))
    for _ in range(args.recall_reps):
        for i in rng.permutation(N):
            ev.append(("recall", int(i)))
        ev.append(("recall", None))
    mixes = [4, 3, 2, 1, 0]                               # electrodes taken from K1 (rest from K2)
    for _ in range(args.blend_reps):
        for n1 in rng.permutation(mixes):
            ev.append(("blend", int(n1)))
    return ev


def blend_channels(n_from_k1: int) -> list[int]:
    return KEYS[1][:n_from_k1] + KEYS[2][: 4 - n_from_k1]


def main() -> None:
    args = parse_args()
    weights_path = os.path.join(tempfile.mkdtemp(prefix="attention_kv_"), "W.npy")
    gap_lead_us = BASE_LEAD_US + int(round(args.pair_gap_ms * 1000 / 40)) * 40

    config = {
        "random_seed": args.seed,
        "coupling_gain_mV": 0.0,       # no K->V propagation until STDP writes it
        "plasticity": True,
        "stdp": True,
        "stdp_unsupervised": True,     # eligibility -> weights directly, no reward
        "weight_max_mV": args.weight_max,
        "plasticity_lr": args.lr,
        "stdp_ltd_ratio": args.ltd_ratio,
        "background_drive_std_mV": args.bg_drive,
        "weights_dump_path": weights_path,  # read back after the run for the learned K->V block
    }
    config.update(json.loads(args.sim_config))
    if cl.is_simulator():
        cl.sim.set_simulator_data_source("organoid_simulator.lif_data_source:make_lif_source", config=config)

    schedule = build_schedule(args)
    ticks_per_second = 200                                       # 5 ms ticks
    ticks_per_event = max(1, round(args.spacing_ms / 1000 * ticks_per_second))
    total_s = (len(schedule) + 2) * ticks_per_event / ticks_per_second

    events: list[tuple[str, object, int]] = []   # (phase, payload, tick timestamp)
    spikes: list[tuple[int, int]] = []           # (timestamp, channel)
    order = "V->K (reverse control)" if args.reverse else "K->V"

    print(f"Attention KV: {len(schedule)} events, {order} pairing, gap {args.pair_gap_ms:g} ms "
          f"(lead_time_us {BASE_LEAD_US} / {gap_lead_us}), ~{total_s:.0f} s of culture time")

    with cl.open() as neurons:
        server = getattr(neurons, "_websocket_server", None)
        viewer_url = server.web_url if server is not None else None
        if viewer_url and not args.no_browser:
            webbrowser.open(viewer_url)

        idx = 0
        last_phase = None
        for tick in neurons.loop(ticks_per_second=ticks_per_second, stop_after_seconds=total_s):
            for s in tick.analysis.spikes:
                spikes.append((int(s.timestamp), int(s.channel)))
            if tick.iteration == 0 or tick.iteration % ticks_per_event != 0:
                continue
            if idx >= len(schedule):
                continue
            phase, payload = schedule[idx]
            idx += 1
            if phase != last_phase:
                print(f"  phase: {phase}")
                last_phase = phase
            ts = int(tick.iteration_timestamp)
            events.append((phase, payload, ts))

            if phase in ("baseline", "recall") and payload is not None:
                neurons.stim(cl.ChannelSet(*KEYS[payload]), STIM, lead_time_us=BASE_LEAD_US)
            elif phase == "blend":
                neurons.stim(cl.ChannelSet(*blend_channels(payload)), STIM, lead_time_us=BASE_LEAD_US)
            elif phase == "train":
                first, second = (VALUES, KEYS) if args.reverse else (KEYS, VALUES)
                neurons.stim(cl.ChannelSet(*first[payload]), STIM, lead_time_us=BASE_LEAD_US)
                neurons.stim(cl.ChannelSet(*second[payload]), STIM, lead_time_us=gap_lead_us)

        # ---------------- decode: spikes per group in the post-query window ----
        spikes_arr = np.array(spikes, dtype=np.int64).reshape(-1, 2)
        spikes_arr = spikes_arr[np.argsort(spikes_arr[:, 0])]
        win = int(args.window_ms * FRAMES_PER_MS)
        ch_to_v = {ch: j for j, grp in enumerate(VALUES) for ch in grp}

        def vcounts(ts: int) -> np.ndarray:
            lo, hi = np.searchsorted(spikes_arr[:, 0], [ts, ts + win])
            out = np.zeros(N)
            for ch in spikes_arr[lo:hi, 1]:
                j = ch_to_v.get(int(ch))
                if j is not None:
                    out[j] += 1
            return out

        def kfired(ts: int, i: int) -> float:
            lo, hi = np.searchsorted(spikes_arr[:, 0], [ts, ts + win])
            return float(np.isin(spikes_arr[lo:hi, 1], KEYS[i]).sum())

        acc: dict[tuple[str, object], list[np.ndarray]] = defaultdict(list)
        kfire: dict[int, list[float]] = defaultdict(list)
        for phase, payload, ts in events:
            if phase == "train":
                continue
            acc[(phase, payload)].append(vcounts(ts))
            if phase in ("baseline", "recall") and payload is not None:
                kfire[payload].append(kfired(ts, payload))

        def matrix(phase: str) -> np.ndarray:
            return np.array([np.mean(acc[(phase, i)], axis=0) for i in range(N)])

        def print_matrix(title: str, m: np.ndarray, spont: np.ndarray) -> None:
            print(f"\n{title}  (mean spikes per query in V group, {args.window_ms:g} ms window)")
            print("          " + "".join(f"{'V' + str(j):>7}" for j in range(N)))
            for i in range(N):
                print(f"  query K{i}" + "".join(f"{m[i, j]:7.2f}" for j in range(N)))
            print("  no stim " + "".join(f"{spont[j]:7.2f}" for j in range(N)))

        def diag_ratio(m: np.ndarray) -> tuple[float, float, float]:
            d = float(np.mean(np.diag(m)))
            off = float(np.mean(m[~np.eye(N, dtype=bool)]))
            return d, off, d / off if off > 0 else float("inf")

        base, rec = matrix("baseline"), matrix("recall")
        print("\n" + "=" * 64)
        print(f"RESULTS  ({order}, seed {args.seed}, {args.train_reps} pairings per key)")
        print("=" * 64)
        print("Key spikes per query (4 electrodes): "
              + ", ".join(f"K{i}={np.mean(kfire[i]):.2f}" for i in range(N)))
        print_matrix("1. BASELINE (before training)", base, np.mean(acc[("baseline", None)], axis=0))
        print_matrix("3. RECALL (after training)", rec, np.mean(acc[("recall", None)], axis=0))

        bd, bo, _ = diag_ratio(base)
        rd, ro, rr = diag_ratio(rec)
        print(f"\n  baseline diagonal {bd:.2f}  off-diagonal {bo:.2f}")
        print(f"  recall   diagonal {rd:.2f}  off-diagonal {ro:.2f}  ratio {rr:.1f}x")
        correct = sum(int(np.argmax(rec[i]) == i and rec[i, i] > 0) for i in range(N))
        print(f"  argmax readout correct for {correct}/{N} keys")

        print("\n4. BLENDED QUERY  (electrodes from K1 / K2 -> V1 vs V2 output)")
        print("   mix      V1      V2   V1 share")
        for n1 in [4, 3, 2, 1, 0]:
            m = np.mean(acc[("blend", n1)], axis=0)
            tot = m[1] + m[2]
            share = f"{m[1] / tot:6.0%}" if tot > 0 else "   n/a"
            print(f"   {n1}/{4 - n1}  {m[1]:6.2f}  {m[2]:6.2f}   {share}")

        if viewer_url and not args.no_hold:
            print(f"\nViewer still running at {viewer_url}")
            input("Press Enter to close it and exit ...")

    # The simulator dumps W at close: W[post, pre]. Show the learned K->V synapses.
    if os.path.exists(weights_path):
        W = np.load(weights_path)
        print("\nLearned weights K_i -> V_j  (mean synapse, mV; read from the simulator, not used in decoding)")
        print("          " + "".join(f"{'V' + str(j):>7}" for j in range(N)))
        for i in range(N):
            print(f"  K{i}      " + "".join(f"{W[np.ix_(VALUES[j], KEYS[i])].mean():7.2f}" for j in range(N)))


if __name__ == "__main__":
    main()
