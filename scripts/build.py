import json
import re
import struct
from collections import Counter
from html import escape
from pathlib import Path

import pyarrow.parquet as pq

root = Path(__file__).resolve().parent.parent
data_dir = root / "data"
out = root / "index.html"
SPLITS = ("train", "validation", "test")
COLORS = {
    "PERSON": "#e8590c", "EMAIL": "#1971c2", "PHONE": "#2b8a3e",
    "ADDRESS": "#9c36b5", "ID_NUMBER": "#c92a2a", "DATE_OF_BIRTH": "#0b7285",
}
FONT = 20


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


records, label_counts, type_counts, problems = [], Counter(), Counter(), []
for split in SPLITS:
    for r in pq.read_table(data_dir / f"{split}-00000-of-00001.parquet").to_pylist():
        if not re.fullmatch(r"images/doc-\d{6}\.png", r["image"]):
            problems.append(f"{r['id']}: unexpected image path")
            continue
        w, h = int(r["width"]), int(r["height"])
        if png_size(root / r["image"]) != (w, h):
            problems.append(f"{r['id']}: image size differs from width/height")
        fields = json.loads(r["fields"])
        for fl in fields:
            x0, y0, x1, y1 = fl["bbox"]
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                problems.append(f"{r['id']}: {fl['label']} box outside image")
            label_counts[fl["label"]] += 1
        type_counts[r["doc_type"]] += 1
        records.append({**r, "split": split, "w": w, "h": h, "fields": fields})
records.sort(key=lambda r: r["id"])


def color(label):
    return COLORS.get(label, "#868e96")


def card(r):
    shapes = []
    for fl in r["fields"]:
        x0, y0, x1, y1 = fl["bbox"]
        c, lab = color(fl["label"]), escape(fl["label"])
        tw, th = len(fl["label"]) * FONT * 0.62 + 10, FONT + 6
        tx, ty = x1 + 8, (y0 + y1 - th) / 2
        if tx + tw > r["w"]:
            tx = x0 - tw - 8
        if tx < 0:
            tx, ty = x0 - 3, max(y0 - th - 4, 0)
        shapes.append(
            f'<rect x="{x0 - 3}" y="{y0 - 3}" width="{x1 - x0 + 6}" height="{y1 - y0 + 6}" fill="{c}" fill-opacity=".12" stroke="{c}" stroke-width="3"/>'
            f'<rect x="{tx:.0f}" y="{ty:.0f}" width="{tw:.0f}" height="{th}" rx="3" fill="{c}"/>'
            f'<text x="{tx + 5:.0f}" y="{ty + FONT - 1:.0f}" font-size="{FONT}" fill="#fff">{lab}</text>'
        )
    rows = "".join(
        f'<li><span class="sw" style="background:{color(fl["label"])}"></span><b>{escape(fl["label"])}</b> {escape(str(fl["text"]))}</li>'
        for fl in r["fields"]
    )
    rid = escape(r["id"])
    return (
        f'<figure class="card"><div class="frame">'
        f'<img src="{escape(r["image"])}" width="{r["w"]}" height="{r["h"]}" loading="lazy" alt="{rid}, synthetic {escape(r["doc_type"])}">'
        f'<svg class="boxes" viewBox="0 0 {r["w"]} {r["h"]}" aria-hidden="true">{"".join(shapes)}</svg></div>'
        f'<figcaption><div class="meta"><b>{rid}</b> · {escape(r["doc_type"])} · {r["split"]} · {len(r["fields"])} fields</div>'
        f"<ul>{rows}</ul></figcaption></figure>"
    )


legend = "".join(
    f'<span><span class="sw" style="background:{color(lab)}"></span>{escape(lab)} <i>{n}</i></span>'
    for lab, n in label_counts.most_common()
)
types = " · ".join(f"{n} {escape(t.replace('_', ' '))}s" for t, n in type_counts.most_common())
splits = Counter(r["split"] for r in records)
split_text = " / ".join(f"{splits[s]} {s}" for s in SPLITS)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PII Document Images</title>
<style>
:root {{ --bg:#f6f6f3; --panel:#fff; --ink:#1c1c1a; --muted:#6a6a64; --line:#e2e1dc; --accent:#3550c8; color-scheme:light; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#141413; --panel:#1d1d1b; --ink:#ebeae4; --muted:#9b9a93; --line:#2f2f2b; --accent:#95a6ff; color-scheme:dark; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ padding:20px 24px 8px; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
p {{ margin:0 0 4px; color:var(--muted); }}
a {{ color:var(--accent); }}
.bar {{ display:flex; flex-wrap:wrap; gap:6px 16px; align-items:center; padding:10px 24px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--bg); z-index:1; font-size:13px; }}
.bar i {{ font-style:normal; color:var(--muted); }}
.bar label {{ margin-left:auto; display:flex; gap:6px; align-items:center; cursor:pointer; }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
main {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; padding:16px 24px 32px; }}
.card {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.frame {{ position:relative; border-bottom:1px solid var(--line); background:#fff; }}
.frame img {{ display:block; width:100%; height:auto; }}
.boxes {{ position:absolute; inset:0; width:100%; height:100%; font-family:system-ui,sans-serif; font-weight:600; }}
body:has(#show:not(:checked)) .boxes {{ display:none; }}
figcaption {{ padding:10px 12px 12px; font-size:12.5px; }}
.meta {{ color:var(--muted); margin-bottom:4px; }}
.meta b {{ color:var(--ink); }}
ul {{ list-style:none; margin:0; padding:0; }}
li {{ padding:2px 0; overflow-wrap:anywhere; }}
li b {{ font-weight:600; font-size:11px; color:var(--muted); margin-right:4px; }}
#show:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
@media (max-width:600px) {{ header, .bar {{ padding-left:16px; padding-right:16px; }} main {{ padding:16px; }} }}
</style>
</head>
<body>
<header>
<h1>PII document images</h1>
<p>{len(records)} synthetic document images ({types}; {split_text}) with labelled PII bounding boxes. CC0-1.0.</p>
<p>Source: <a href="https://huggingface.co/datasets/hussain-khalid/pii-dataset" target="_blank" rel="noopener noreferrer">https://huggingface.co/datasets/hussain-khalid/pii-dataset</a></p>
</header>
<div class="bar">{legend}<label><input type="checkbox" id="show" checked> Show boxes</label></div>
<main>
{"".join(card(r) for r in records)}
</main>
</body>
</html>
"""
out.write_text(html, encoding="utf-8", newline="\n")
print(f"images={len(records)} splits={dict(splits)} types={dict(type_counts)} labels={dict(label_counts)}")
print(f"problems={len(problems)}", *problems[:10], sep="\n")
print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")
