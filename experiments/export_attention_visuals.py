"""Run the reservoir attention experiment (stock simulator) and export raw data for visuals.

Data for docs/attention-visuals. Run from the repo root:
  CL_SDK_ACCELERATED_TIME=1 uv run python experiments/export_attention_visuals.py viz.json
"""
import json
import sys

sys.path.insert(0, "experiments")
import numpy as np
import attention_reservoir as a

SEED, TRIALS, REPEATS = 42, 8, 40
out_path = sys.argv[1]

rng = np.random.default_rng(SEED)
queries, keys = a.make_trials(TRIALS, rng)
delta = round(5.0 * a.FRAMES_PER_MS)
pres = []
for q, k in zip(queries, keys):
    for _ in range(REPEATS):
        pres.append(a.Presentation(k == 1, np.broadcast_to(q == 1, (a.SLOTS, a.DIMS)).copy(), 0.5, 0.5, delta))

spikes, stims = a.run_presentations(pres, seed=SEED, coupling=0.0)
a.check_timing(pres, stims)
ts, ch = spikes
W = round(a.WINDOW_MS * a.FRAMES_PER_MS)          # readout window after the query pulse (2 ms)
PRE, POST = -5 * a.FRAMES_PER_MS, 30 * a.FRAMES_PER_MS

trials = []
cat_hits = {"neither": [0, 0], "key": [0, 0], "query": [0, 0], "both": [0, 0]}   # [spiked, total]
for t in range(TRIALS):
    q, k = queries[t], keys[t]
    counts = np.zeros((a.SLOTS, a.DIMS), dtype=int)
    raster = []          # per presentation: list of [slot, dim, ms after key pulse]
    for r in range(REPEATS):
        p = pres[t * REPEATS + r]
        c = a.counts_in_window(spikes, p.t_query, p.t_query + W)[a.ALL_CHANNELS]
        counts += c
        for i in range(a.SLOTS):
            for j in range(a.DIMS):
                name = {(0, 0): "neither", (1, 0): "key", (0, 1): "query", (1, 1): "both"}[(int(k[i, j]), int(q[j]))]
                cat_hits[name][0] += int(c[i, j] > 0)
                cat_hits[name][1] += 1
        lo, hi = np.searchsorted(ts, [p.t_key + PRE, p.t_key + POST])
        pts = []
        for s_ts, s_ch in zip(ts[lo:hi].tolist(), ch[lo:hi].tolist()):
            if 8 <= s_ch <= 39:
                pts.append([(s_ch - 8) // 8, (s_ch - 8) % 8, round((s_ts - p.t_key) / a.FRAMES_PER_MS, 2)])
        raster.append(pts)
    slot = counts.sum(1)
    qk = (k @ q)
    trials.append({
        "q": q.tolist(), "keys": k.tolist(), "qk": qk.tolist(),
        "counts": counts.tolist(), "slot": slot.tolist(),
        "pick": int(slot.argmax()), "best": int(qk.argmax()),
        "tie": bool((slot == slot.max()).sum() > 1),
        "raster": raster,
    })

json.dump({"seed": SEED, "repeats": REPEATS, "delta_ms": 5.0, "window_ms": a.WINDOW_MS,
           "amp_uA": 0.5, "categories": cat_hits, "trials": trials}, open(out_path, "w"))
print("trials:", [(tr["slot"], tr["qk"], tr["pick"] == tr["best"]) for tr in trials])
print("P(spike in readout window):", {k: round(v[0] / v[1], 3) for k, v in cat_hits.items()})
