#!/usr/bin/env python3
"""Build normalized, transparent animation frames for the Educational Studio.

Usage:  python3 tools/build_frames.py [repo_root] [qa_output_dir]

- Splits every sheet in poses/ into single figures (connected components).
- Removes the white background (soft alpha, no white halo).
- Scales every figure so feet-to-head-top == BH, so all clips share one scale.
- Writes frames/<group>/<name>.webp on one common canvas (feet on the bottom edge,
  body anchor on the horizontal centre) and embeds the clip list into index.html.
"""
import json, os, re, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
QA = sys.argv[2] if len(sys.argv) > 2 else None
BH = 480        # body height (feet -> top of head) in canvas pixels
PAD = 8
PAD_B = 6

# ---- sources -------------------------------------------------------------
GEST = "poses/gestures/"
POSES = {  # name: (file, side) side 0 = left figure, 1 = right figure
    "idle": ("05-idle-hand-across.webp", 1),
    "hand_across": ("08-standing-hand-across.webp", 1),
    "chest": ("01-hand-chest-wave.webp", 0),
    "wave_hi": ("06-wave-high-low.webp", 0),
    "wave_lo": ("06-wave-high-low.webp", 1),
    "point_l": ("03-point-left-right.webp", 0),
    "point_r": ("03-point-left-right.webp", 1),
    "cupped": ("04-cupped-hands-stop.webp", 0),
    "stop": ("04-cupped-hands-stop.webp", 1),
    "open": ("02-open-hands-chest.webp", 0),
    "one": ("09-one-two-fingers.webp", 0),
    "two": ("09-one-two-fingers.webp", 1),
    "offer": ("10-offer-hands.webp", 0),
    "open2": ("10-offer-hands.webp", 1),
    "explain": ("12-explain-pinch.webp", 0),
    "pinch": ("12-explain-pinch.webp", 1),
    "point_up_r": ("13-point-up-right.webp", 0),
    "point_r2": ("13-point-up-right.webp", 1),
    "chin": ("15-chin-temple.webp", 0),
    "temple": ("15-chin-temple.webp", 1),
    "point_r4": ("16-point-present.webp", 0),
    "present": ("16-point-present.webp", 1),
    "think": ("17-chin-think.webp", 0),
    "think_up": ("17-chin-think.webp", 1),
}
WALKS = {  # group: (sheet, anchor mode)
    "walk-side": ("poses/walk-run/10-walk-side-cycle.png", "torso"),
    "walk-back": ("poses/walk-run/08-walk-back-cycle.png", "torso"),
}


def load_figures(path, n):
    im = np.asarray(Image.open(os.path.join(ROOT, path)).convert("RGB")).astype(np.float32)
    d = (255 - im).max(axis=2)
    alpha = np.clip((d - 12) / 28.0, 0, 1)
    a3 = alpha[..., None]
    rgb = np.where(a3 > 0.01, (im - 255 * (1 - a3)) / np.maximum(a3, 0.01), 0).clip(0, 255)
    mask = d > 25
    lab, k = ndi.label(mask, structure=np.ones((3, 3)))
    areas = ndi.sum(mask, lab, range(1, k + 1))
    big = [i + 1 for i in np.argsort(areas)[::-1][:n]]
    cent = {b: ndi.center_of_mass(mask, lab, b)[1] for b in big}
    owner = {}
    for i in range(1, k + 1):
        if i in big:
            owner[i] = i
        else:
            c = ndi.center_of_mass(mask, lab, i)[1]
            owner[i] = min(big, key=lambda b: abs(cent[b] - c))
    figs = []
    for b in sorted(big, key=lambda b: cent[b]):
        own = ndi.binary_dilation(np.isin(lab, [i for i, o in owner.items() if o == b]), iterations=2)
        a_full = alpha * own
        ys = np.where((a_full > 0.15).any(axis=1))[0]
        xs = np.where((a_full > 0.15).any(axis=0))[0]
        y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
        figs.append((a_full[y0:y1, x0:x1], rgb[y0:y1, x0:x1]))
    return figs


def measure(a, mode):
    """Return (anchor_x, head_top_y) in source pixels."""
    h, w = a.shape
    if mode == "legs":
        rows = a[int(h * 0.65):int(h * 0.95)]
    else:
        rows = a[int(h * 0.2):int(h * 0.5)]
    cx = (rows.sum(axis=0) * np.arange(w)).sum() / max(rows.sum(), 1)
    half = max(20, int(0.09 * h))
    lo, hi = max(0, int(cx) - half), min(w, int(cx) + half)
    ys = np.where((a[:, lo:hi] > 0.3).any(axis=1))[0]
    return cx, float(ys[0])


def normalize(a, c, mode):
    cx, top = measure(a, mode)
    body = a.shape[0] - top
    s = BH / body
    rgba = np.dstack([c, a * 255]).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA")
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    img = img.resize((nw, nh), Image.LANCZOS)
    return img, cx * s, top * s


# ---- collect all frames ---------------------------------------------------
frames = {}  # key -> (img, cx, top)
cache = {}
for name, (file, side) in POSES.items():
    if file not in cache:
        cache[file] = load_figures(GEST + file, 2)
    a, c = cache[file][side]
    frames["pose/" + name] = normalize(a, c, "legs")
for group, (sheet, mode) in WALKS.items():
    for i, (a, c) in enumerate(load_figures(sheet, 8)):
        frames[f"{group}/{i + 1:02d}"] = normalize(a, c, mode)

# ---- common canvas -----------------------------------------------------------
half = max(max(cx, img.width - cx) for img, cx, top in frames.values())
up = max(top for img, cx, top in frames.values())
CW = 2 * (int(np.ceil(half)) + PAD)
CH = BH + int(np.ceil(up)) + PAD + PAD_B
print("canvas", CW, CH, "frames", len(frames))

out_dir = os.path.join(ROOT, "frames")
for old in ("walk-side", "walk-back", "pose", "previews"):
    p = os.path.join(out_dir, old)
    if os.path.isdir(p):
        for f in os.listdir(p):
            os.remove(os.path.join(p, f))
rendered = {}
for key, (img, cx, top) in frames.items():
    canvas = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    ox = int(round(CW / 2 - cx))
    oy = CH - PAD_B - img.height
    canvas.paste(img, (ox, oy), img)
    path = os.path.join(out_dir, key + ".webp")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path, "WEBP", quality=90, alpha_quality=100, method=6)
    rendered[key] = canvas

# ---- clips ---------------------------------------------------------------------
def seq(*items):
    return [["pose/" + n + ".webp", ms] for n, ms in items]


def cycle(group, ms):
    return [[f"{group}/{i:02d}.webp", ms] for i in range(1, 9)]


CLIPS = {
    "idle": dict(name="وقوف (تنفّس)", fade=0, steps=seq(("idle", 1000))),
    "wave": dict(name="تحية بالتلويح", fade=70, steps=seq(
        ("idle", 500), ("wave_hi", 260), ("wave_lo", 260), ("wave_hi", 260), ("wave_lo", 260),
        ("wave_hi", 260), ("wave_lo", 260), ("idle", 700))),
    "greet": dict(name="تحية باليد على الصدر", fade=160, steps=seq(("idle", 600), ("chest", 1400), ("idle", 600))),
    "talk": dict(name="يشرح بيديه", fade=200, steps=seq(
        ("idle", 500), ("hand_across", 700), ("open", 700), ("explain", 700), ("pinch", 700),
        ("offer", 700), ("open2", 700), ("idle", 500))),
    "point_right": dict(name="يشير لليمين", fade=160, steps=seq(("idle", 500), ("point_r", 1600), ("idle", 400))),
    "point_left": dict(name="يشير لليسار", fade=160, steps=seq(("idle", 500), ("point_l", 1600), ("idle", 400))),
    "point_up": dict(name="يشير للأعلى", fade=160, steps=seq(("idle", 500), ("point_up_r", 1500), ("point_r2", 1200), ("idle", 400))),
    "present": dict(name="يقدّم ويعرض", fade=180, steps=seq(("idle", 400), ("present", 1600), ("point_r4", 1400), ("idle", 400))),
    "count": dict(name="يعدّ بالأصابع", fade=100, steps=seq(("idle", 500), ("one", 1100), ("two", 1100), ("idle", 500))),
    "stop": dict(name="انتبهوا (إيقاف)", fade=120, steps=seq(("idle", 400), ("cupped", 800), ("stop", 1400), ("idle", 400))),
    "think": dict(name="يفكّر", fade=200, steps=seq(
        ("idle", 400), ("think", 1200), ("think_up", 1400), ("chin", 1200), ("temple", 1200), ("idle", 400))),
    "walk_side": dict(name="مشي جانبي", face="right", walk=True, fade=0, steps=cycle("walk-side", 100)),
    "walk_back": dict(name="مشي من الخلف", walk=True, fade=0, steps=cycle("walk-back", 100)),
}
for c in CLIPS.values():
    c.setdefault("loop", True)

manifest = {"FRM": {"BH": BH, "CW": CW, "CH": CH}, "CLIPS": CLIPS}
with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

# embed into index.html (so it also works when opened from disk)
idx = os.path.join(ROOT, "index.html")
html = open(idx, encoding="utf-8").read()
blob = "window.FRM=%s;window.CLIPS=%s;" % (
    json.dumps(manifest["FRM"]), json.dumps(CLIPS, ensure_ascii=False, separators=(",", ":")))
new, n = re.subn(r"/\*CLIPS_START\*/.*?/\*CLIPS_END\*/",
                 lambda m: "/*CLIPS_START*/" + blob + "/*CLIPS_END*/", html, flags=re.S)
if n:
    open(idx, "w", encoding="utf-8").write(new)
    print("embedded clips into index.html")
else:
    print("WARNING: CLIPS markers not found in index.html")

# ---- QA contact sheets (not committed) -------------------------------------------
if QA:
    os.makedirs(QA, exist_ok=True)
    def sheet(keys, path, cols, scale=0.38):
        tw, th = int(CW * scale), int(CH * scale)
        rows = -(-len(keys) // cols)
        im = Image.new("RGBA", (tw * cols, (th + 14) * rows), (150, 160, 170, 255))
        dr = ImageDraw.Draw(im)
        for i, k in enumerate(keys):
            t = rendered[k].resize((tw, th), Image.LANCZOS)
            x, y = (i % cols) * tw, (i // cols) * (th + 14)
            im.alpha_composite(t, (x, y + 14))
            dr.text((x + 3, y + 1), k, fill=(255, 255, 255, 255))
            dr.line([(x, y + 14 + int(th - PAD_B * scale)), (x + tw, y + 14 + int(th - PAD_B * scale))], fill=(255, 80, 80, 255))
        im.convert("RGB").save(path)
    sheet([k for k in rendered if k.startswith("pose/")], os.path.join(QA, "qa_poses.png"), 8)
    sheet([k for k in rendered if k.startswith("walk-side")], os.path.join(QA, "qa_walk_side.png"), 8)
    sheet([k for k in rendered if k.startswith("walk-back")], os.path.join(QA, "qa_walk_back.png"), 8)
    for cid, clip in (("walk_side", CLIPS["walk_side"]), ("walk_back", CLIPS["walk_back"]), ("wave", CLIPS["wave"])):
        fr = []
        for f, ms in clip["steps"]:
            bg = Image.new("RGBA", (CW, CH), (150, 160, 170, 255))
            bg.alpha_composite(rendered[f[:-5]])
            fr.append((bg.convert("RGB"), ms))
        fr[0][0].save(os.path.join(QA, f"qa_{cid}.gif"), save_all=True, append_images=[x[0] for x in fr[1:]],
                      duration=[x[1] for x in fr], loop=0)
    print("qa written to", QA)
