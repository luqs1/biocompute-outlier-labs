"""Does the STOCK (main-branch) LIF simulator learn from stimulation alone?
Drives it exactly as the SDK does: on_stims() + read(). No apply_reward."""
import importlib.util, sys
from types import SimpleNamespace
import numpy as np
spec = importlib.util.spec_from_file_location("stock_lif", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
A, B = [8, 9, 16, 17], [40, 41, 48, 49]
PERIOD, LAG, BATCH = 2500, 125, 500
def stim(ch, ts, a=2.0):
    return SimpleNamespace(timestamp=ts, channel=ch, phase_durations_us=(200, 200), phase_currents_uA=(-a, a))
def train(src, ts, n):
    for _ in range(n):
        t0 = ts + 100
        src.on_stims([stim(c, t0) for c in A] + [stim(c, t0 + LAG) for c in B])
        for _ in range(PERIOD // BATCH):
            src.read(ts, BATCH); ts += BATCH
    return ts
def probe(src, ts, n=30):
    hits = 0
    for _ in range(n):
        t0 = ts + 100
        src.on_stims([stim(c, t0) for c in A])
        for _ in range(PERIOD // BATCH):
            b = src.read(ts, BATCH)
            hits += sum(1 for s in b.spikes if s.channel in B and t0 < s.timestamp <= t0 + 750)
            ts += BATCH
    return hits / n, ts
cfgs = {
  "defaults (.env)":               {},
  "plasticity":                    {"plasticity": True},
  "plasticity+stdp":               {"plasticity": True, "stdp": True},
  "coupling 4 + plasticity+stdp":  {"coupling_gain_mV": 4.0, "plasticity": True, "stdp": True},
  "coupling 4 (no plasticity)":    {"coupling_gain_mV": 4.0},
}
print(f"{'config':32s} {'W changed?':>12s} {'B per A-probe before':>22s} {'after 200 pairings':>20s}")
for name, cfg in cfgs.items():
    src = m.LIFDataSource(random_seed=42, **cfg); src.open()
    try: W0 = src.coupling_weights
    except AssertionError: W0 = None
    before, ts = probe(src, 0)
    ts = train(src, ts, 200)
    after, ts = probe(src, ts)
    try: W1 = src.coupling_weights
    except AssertionError: W1 = None
    dW = "no coupling" if W0 is None else f"{np.abs(W1 - W0).max():.4f}"
    print(f"{name:32s} {dW:>12s} {before:22.2f} {after:20.2f}")
    src.close()
