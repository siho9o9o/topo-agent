# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hard sample walls (#129) for hold finding.

Draw scenes that would be hard in a real gym (not photos of a real gym, so they may be sent to outside AI).
Each scene places problem holds (the answer) and other problems' holds (distractors).

- light   uneven light (one side dark, lit spots blown out whitish)
- volume  holds on wooden volumes (big protrusions) and in their shadows
- similar mixed similar-color problems (orange and pink next to a red problem)
- black   black holds (bright wall)
- white   white holds (dark wall)
- gray    gray holds (bright wall)
- twins   another problem of the same color far away (on the next wall; checks over-picking)
- busy    tiny dots of the problem color all over the wall (tape scraps, screws, dirt. The over-picked shape from a real photo)
- tape    same-color tape on variously colored holds (color cannot tell. Only the AI's eyes can)

    python -m topo_agent.eval --bench --methods color
"""
import io
import math
import random

from PIL import Image, ImageDraw, ImageFilter


def _hold_shape(rnd, cx, cy, r):
    """A bumpy outline (a hold-like shape). From server/demo.py of mirai-climbing-v2."""
    n = rnd.randint(7, 11)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rnd.uniform(-0.2, 0.2)
        rr = r * rnd.uniform(0.65, 1.15)
        pts.append((cx + rr * math.cos(a), cy + rr * 0.8 * math.sin(a)))
    return pts


WIDTH, HEIGHT = 1200, 900
PER_PROBLEM = 7
NAMES = ("light", "volume", "similar", "black", "white", "gray", "twins", "busy", "tape")
BEIGE, DARK = (214, 206, 192), (52, 52, 56)
OTHERS = [(205, 40, 45), (35, 85, 200), (240, 200, 30), (40, 150, 70), (130, 60, 170)]


def _wall(rnd, base):
    img = Image.new("RGB", (WIDTH, HEIGHT), base)
    d = ImageDraw.Draw(img)
    for _ in range(5000):  # wall grain
        g = rnd.randint(-12, 12)
        d.point((rnd.randrange(WIDTH), rnd.randrange(HEIGHT)), fill=tuple(min(255, max(0, c + g)) for c in base))
    seam = tuple(max(0, c - 24) for c in base)
    for x in range(0, WIDTH, 240):
        d.line([(x, 0), (x, HEIGHT)], fill=seam, width=2)
    for y in range(0, HEIGHT, 240):
        d.line([(0, y), (WIDTH, y)], fill=seam, width=2)
    for y in range(40, HEIGHT, 80):  # bolt holes
        for x in range(40, WIDTH, 80):
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=tuple(max(0, c - 90) for c in base))
    return img.filter(ImageFilter.SMOOTH)


def _place(rnd, placed, n, box, rmin=18, rmax=40):
    """Pick n non-overlapping placements inside box (x0, y0, x1, y1)."""
    out = []
    for _ in range(3000):
        if len(out) >= n:
            break
        r = rnd.uniform(rmin, rmax)
        cx, cy = rnd.uniform(box[0] + r, box[2] - r), rnd.uniform(box[1] + r, box[3] - r)
        if any((cx - x) ** 2 + (cy - y) ** 2 < (r + rr + 26) ** 2 for x, y, rr in placed):
            continue
        placed.append((cx, cy, r))
        out.append((cx, cy, r, _hold_shape(rnd, cx, cy, r)))
    return out


def _draw_hold(d, pts, color, cx, cy, r, shadow=(150, 144, 132)):
    d.polygon([(x + 4, y + 5) for x, y in pts], fill=shadow)
    d.polygon(pts, fill=color)
    light = tuple(min(255, c + 40) for c in color)
    d.ellipse([cx - r * 0.35, cy - r * 0.45, cx - r * 0.05, cy - r * 0.15], fill=light)


def _shade(img, factor):
    """Change brightness per pixel (multiply by factor(x, y)). For uneven light and shadows."""
    px = img.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            f = factor(x, y)
            if f != 1:
                r, g, b = px[x, y]
                if f > 1:  # pull bright spots toward white (blown out)
                    t = min(f - 1, 1)
                    px[x, y] = tuple(int(c + (255 - c) * t) for c in (r, g, b))
                else:
                    px[x, y] = tuple(int(c * f) for c in (r, g, b))


def scene(name, seed=11):
    """One scene. Returns {"name", "jpeg", "truth": [{"points": fractional points}], "tap": {"x", "y"}}."""
    rnd = random.Random(f"{name}-{seed}")
    wall = DARK if name == "white" else BEIGE
    img = _wall(rnd, wall)
    d = ImageDraw.Draw(img)
    placed = []
    volumes = []
    if name == "volume":  # two wooden volumes (drawn before the holds)
        for vx, vy in ((360, 420), (820, 330)):
            pts = [(vx - 170, vy + 120), (vx - 40, vy - 150), (vx + 160, vy - 60), (vx + 120, vy + 150)]
            d.polygon([(x + 18, y + 22) for x, y in pts], fill=(120, 114, 104))
            d.polygon(pts, fill=(196, 160, 110))
            d.polygon([pts[0], pts[1], (vx, vy)], fill=(150, 120, 80))  # shaded face
            volumes.append(pts)
    target = {"light": (40, 150, 70), "volume": (130, 60, 170), "similar": (205, 40, 45), "black": (28, 28, 30),
              "white": (240, 240, 236), "gray": (128, 128, 128), "twins": (35, 85, 200), "busy": (205, 40, 45),
              "tape": None}[name]
    route = _place(rnd, placed, PER_PROBLEM, (520, 60, 1160, 840) if name == "twins" else (300, 60, 900, 840))
    if name == "twins":  # another problem of the same color at the left edge (about 30% of the photo width from the right problem)
        for cx, cy, r, pts in _place(rnd, placed, 5, (40, 60, 160, 840), 16, 30):
            _draw_hold(d, pts, target, cx, cy, r)
    others = [c for c in OTHERS if c != target]
    if name == "similar":
        others = [(235, 120, 30), (230, 80, 160)] * 2  # orange/pink
    if name in ("black", "white", "gray"):
        others = OTHERS[:3]
    rest = _place(rnd, placed, 22, (40, 40, WIDTH - 40, HEIGHT - 40))
    tape = (250, 250, 250)
    shadow = (30, 30, 32) if name == "white" else (150, 144, 132)
    for i, (cx, cy, r, pts) in enumerate(rest):
        _draw_hold(d, pts, others[i % len(others)], cx, cy, r, shadow)
    for i, (cx, cy, r, pts) in enumerate(route):
        color = OTHERS[i % len(OTHERS)] if name == "tape" else target
        _draw_hold(d, pts, color, cx, cy, r, shadow)
        if name == "tape":  # a white tape scrap under the hold
            d.rectangle([cx - 14, cy + r * 0.8 + 4, cx + 14, cy + r * 0.8 + 12], fill=tape)
    if name == "busy":  # tiny dots of the problem color (far smaller than holds)
        for _ in range(90):
            x, y, r = rnd.uniform(20, WIDTH - 20), rnd.uniform(20, HEIGHT - 20), rnd.uniform(2, 6)
            if any((x - cx) ** 2 + (y - cy) ** 2 < (rr + r + 8) ** 2 for cx, cy, rr, _ in route):
                continue
            d.ellipse([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=target)
    if name == "light":  # dark on the left, a light at the top right
        _shade(img, lambda x, y: (0.45 + 0.55 * x / WIDTH)
               + 0.9 * max(0.0, 1 - math.hypot(x - 950, y - 150) / 260))
    if name == "volume":  # shadows falling under the volumes
        _shade(img, lambda x, y: 0.6 if 300 < x < 520 and 560 < y < 700 else 1)
    plain = route[1:-1] or route
    cx, cy, _, _ = plain[rnd.randrange(len(plain))]
    img = img.filter(ImageFilter.GaussianBlur(1.2))  # blur slightly and save coarsely, like a phone photo
    out = io.BytesIO()
    img.save(out, "JPEG", quality=70)
    return {"name": name, "jpeg": out.getvalue(), "tap": {"x": cx / WIDTH, "y": cy / HEIGHT},
            "truth": [{"points": [[x / WIDTH, y / HEIGHT] for x, y in pts], "x": c / WIDTH, "y": e / HEIGHT, "r": r / WIDTH}
                      for c, e, r, pts in route]}


def items(names=NAMES, seeds=(11, 12)):
    """A shape passable to hold_eval.evaluate (as problem marks)."""
    out = []
    for name in names:
        for seed in seeds:
            s = scene(name, seed)
            marks = [{"kind": "hold", "use": "any", "x": t["x"], "y": t["y"], "r": t["r"], "points": t["points"]}
                     for t in s["truth"]]
            out.append({"id": f"{name}-{seed}", "jpeg": s["jpeg"], "tap": s["tap"],
                        "marks": {"marks": marks, "photo": {"width": WIDTH, "height": HEIGHT}}})
    return out
