"""Direct (no SDK loop) check of LIFDataSource(stdp_unsupervised=True).

Run: uv run python tests/test_stdp_unsupervised.py

Pairs group A then group B ~5 ms later. Expect W[B, A] (A->B, causal) to grow
and W[A, B] (B->A, acausal) to stay ~0; reversed order does the opposite.
Then stimulates A alone and counts B spikes before vs after training.
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from organoid_simulator.lif_data_source import LIFDataSource  # noqa: E402

A = [8, 9, 16, 17]
B = [40, 41, 48, 49]
FPS = 25_000
PERIOD_FRAMES = 2_500        # 100 ms between pairings
LAG_FRAMES = 125             # 5 ms A -> B
BATCH = 500                  # 20 ms read batches (like a 50 Hz loop)


def stim(ch, ts, amp_uA=2.0):
    return SimpleNamespace(timestamp=ts, channel=ch,
                           phase_durations_us=(200, 200), phase_currents_uA=(-amp_uA, amp_uA))


def run(first, second, n_pairs, **cfg):
    src = LIFDataSource(random_seed=1, stdp_unsupervised=True, **cfg)
    src.open()
    ts = 0
    for k in range(n_pairs):
        t0 = ts + 100
        src.on_stims([stim(c, t0) for c in first] + [stim(c, t0 + LAG_FRAMES) for c in second])
        for _ in range(PERIOD_FRAMES // BATCH):
            src.read(ts, BATCH)
            ts += BATCH
    return src, ts


def probe(src, ts, n=20):
    """Stim A alone n times; return mean B spikes per probe (within 20 ms)."""
    hits = 0
    for _ in range(n):
        t0 = ts + 100
        src.on_stims([stim(c, t0) for c in A])
        for _ in range(PERIOD_FRAMES // BATCH):
            b = src.read(ts, BATCH)
            hits += sum(1 for s in b.spikes if s.channel in B and t0 < s.timestamp <= t0 + 500)
            ts += BATCH
    return hits / n


def block(W, post, pre):
    return float(W[np.ix_(post, pre)].mean())


def main():
    cfg = dict(plasticity_lr=0.25, weight_max_mV=6.0, stdp_ltd_ratio=1.05)
    n_pairs = int(os.getenv("N_PAIRS", "40"))

    # initial weights
    s0 = LIFDataSource(random_seed=1, stdp_unsupervised=True, **cfg); s0.open()
    W0 = s0.coupling_weights
    print(f"initial W: max={W0.max():.3f}  (coupling_gain_mV=0)")
    base = probe(s0, 0)
    print(f"untrained: B spikes per A-only probe = {base:.2f}")

    with tempfile.TemporaryDirectory() as d:
        dump = os.path.join(d, "W.npy")
        src, ts = run(A, B, n_pairs, weights_dump_path=dump, **cfg)
        W = src.coupling_weights
        mask = np.ones_like(W, bool); mask[np.ix_(B, A)] = False; mask[np.ix_(A, B)] = False
        print(f"A->B order, {n_pairs} pairs: W[B,A]={block(W, B, A):.3f}  W[A,B]={block(W, A, B):.3f}  "
              f"other-mean={W[mask].mean():.4f}  other-max={W[mask].max():.3f}")
        trained = probe(src, ts)
        print(f"trained:   B spikes per A-only probe = {trained:.2f}")
        src.close()
        Wd = np.load(dump)
        assert np.allclose(Wd, src.coupling_weights, atol=1e-5), "dump mismatch"
        print(f"weights dump OK: {dump} shape={Wd.shape}")
        assert block(W, B, A) > 5 * max(block(W, A, B), 1e-3), "causal should dominate"
        assert trained > base + 1.0, "A alone should now drive B"

    src, ts = run(B, A, n_pairs, **cfg)
    W = src.coupling_weights
    print(f"B->A order, {n_pairs} pairs: W[B,A]={block(W, B, A):.3f}  W[A,B]={block(W, A, B):.3f}")
    assert block(W, A, B) > 5 * max(block(W, B, A), 1e-3), "reversed order should flip"

    # default behaviour unchanged: no unsupervised -> W stays zero
    d = LIFDataSource(random_seed=1, plasticity=True, stdp=True); d.open()
    d.on_stims([stim(c, 100) for c in A] + [stim(c, 225) for c in B]); d.read(0, 2500)
    assert d.coupling_weights.max() == 0.0
    print("PASS")


if __name__ == "__main__":
    main()
