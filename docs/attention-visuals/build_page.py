"""Build docs/attention-visuals/index.html.

Run from the repo root:
  CL_SDK_ACCELERATED_TIME=1 uv run python experiments/export_attention_visuals.py viz.json
  python docs/attention-visuals/build_page.py viz.json docs/attention-visuals/viewer-trial1.png docs/attention-visuals/index.html docs/attention-visuals/template.html
"""
import base64, json, sys

viz = json.load(open(sys.argv[1]))
shot = base64.b64encode(open(sys.argv[2], "rb").read()).decode()
out = sys.argv[3]

T = 1  # the example trial (also the one in the viewer screenshot)
tr = viz["trials"][T]
# four example electrodes from trial 1: (slot, dim, label)
picks = [(3, 4, "both bits on", "both"), (2, 4, "query bit only", "query"),
         (1, 1, "key bit only", "key"), (2, 0, "neither bit", "neither")]
raster = {f"{s}-{d}": [[r, p[2]] for r, pres in enumerate(tr["raster"]) for p in pres if p[0] == s and p[1] == d]
          for s, d, _, _ in picks}
data = {
    "example": {k: tr[k] for k in ("q", "keys", "qk", "counts", "slot", "pick", "best")},
    "picks": picks, "raster": raster, "categories": viz["categories"],
    "trials": [{k: t[k] for k in ("qk", "slot", "pick", "best", "tie")} for t in viz["trials"]],
    "repeats": viz["repeats"],
}
html = open(sys.argv[4]).read().replace("__DATA__", json.dumps(data)).replace("__SHOT__", shot)
open(out, "w").write(html)
print("wrote", out, len(html))
