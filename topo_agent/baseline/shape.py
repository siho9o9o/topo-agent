# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tell hold outlines apart by photo color (AI outlines are blocky, and we also want the shape at the tapped spot).

Method (no numpy, only Pillow and plain Python. Runs on a small image about 96 px square per hold):
1. Crop around the hold, shrink it, and take the center color as the "hold color"
2. From the center, follow and spread over pixels of the same color. Saturated holds are compared by hue;
   white/black/gray holds by brightness
   (brightness changes with lighting but hue barely does. Wall shadows and edges are hard to pick up)
3. Fill holes inside the blob (shiny spots etc.), trace the edge, reduce points, and convert to photo fractions
4. When the blob spills far over the crop edge, it connects to the wall's own color, so drop it
"""
import colorsys
import math
from collections import deque

WORK = 96  # long side of the cropped image (pixels)
MAX_POINTS = 32
SCAN = 640  # long side of the photo when finding same-color holds (pixels; keeps small holds catchable)
MAX_EXTRA = 40
MIN_BLOB = 6  # minimum blob size added as a missed hold (pixels of the search image)
BLOB_HUE = 0.035  # upper bound of the whole blob's hue gap from the tapped hold


def _hsv(rgb):
    return colorsys.rgb_to_hsv(*(v / 255 for v in rgb))


def _hue_gap(a, b):
    return min(abs(a - b), 1 - abs(a - b))


def colored(c):
    """Whether it is a saturated hold (not white, black or gray)."""
    return c[1] >= 0.28 and c[2] >= 0.15


def same(c, seed):
    """Whether pixel c looks the same as the hold color seed."""
    if colored(seed):
        # Compare saturation loosely so washed-out highlights are still caught. Leave out dark spots (shadows, ceiling beams)
        return (_hue_gap(c[0], seed[0]) < 0.05 and c[1] >= max(0.2, seed[1] * 0.4)
                and c[2] >= max(0.12, seed[2] * 0.55))
    return c[1] < 0.25 and abs(c[2] - seed[2]) < 0.12


def _mean(colors):
    if not colors:
        return (0, 0, 0)
    # Hue is an angle, so average directions instead of adding values
    x = sum(math.cos(c[0] * 2 * math.pi) * c[1] for c in colors)
    y = sum(math.sin(c[0] * 2 * math.pi) * c[1] for c in colors)
    h = (math.atan2(y, x) / (2 * math.pi)) % 1
    return (h, sum(c[1] for c in colors) / len(colors), sum(c[2] for c in colors) / len(colors))


def _trace(mask, w, h):
    """Trace a blob edge clockwise (Moore-neighborhood tracing)."""
    start = next(((x, y) for y in range(h) for x in range(w) if mask[y][x]), None)
    if not start:
        return []
    dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    out, cur, back = [start], start, 4
    for _ in range(w * h * 4):
        for i in range(8):
            d = (back + 1 + i) % 8
            nx, ny = cur[0] + dirs[d][0], cur[1] + dirs[d][1]
            if 0 <= nx < w and 0 <= ny < h and mask[ny][nx]:
                cur, back = (nx, ny), (d + 4) % 8
                break
        else:
            break
        if cur == start:
            break
        out.append(cur)
    return out


def _simplify(points, tol):
    """Reduce points (Douglas-Peucker). The outline is closed, so split at the two farthest points first."""
    def rdp(pts):
        if len(pts) < 3:
            return pts
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5 or 1
        far, idx = 0, 0
        for i, (x, y) in enumerate(pts[1:-1], 1):
            d = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / norm
            if d > far:
                far, idx = d, i
        if far <= tol:
            return [pts[0], pts[-1]]
        return rdp(pts[:idx + 1])[:-1] + rdp(pts[idx:])
    if len(points) < 4:
        return points
    i = max(range(len(points)), key=lambda k: (points[k][0] - points[0][0]) ** 2 + (points[k][1] - points[0][1]) ** 2)
    return rdp(points[:i + 1])[:-1] + rdp(points[i:] + [points[0]])[:-1]


def _grow(hsv, w, h, sx, sy, seed):
    """The blob of seed-colored pixels traced from (sx, sy)."""
    mask = [[False] * w for _ in range(h)]
    if not same(hsv[sy][sx], seed):
        return mask, 0
    q, mask[sy][sx], count = deque([(sx, sy)]), True, 1
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not mask[ny][nx] and same(hsv[ny][nx], seed):
                mask[ny][nx] = True
                count += 1
                q.append((nx, ny))
    return mask, count


def _fill_holes(mask, w, h):
    """Fill holes inside a blob (shiny white spots etc.). What the edge cannot reach is a hole."""
    outside = [[False] * w for _ in range(h)]
    q = deque((x, y) for y in range(h) for x in range(w) if (x in (0, w - 1) or y in (0, h - 1)) and not mask[y][x])
    for x, y in q:
        outside[y][x] = True
    while q:
        x, y = q.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not mask[ny][nx] and not outside[ny][nx]:
                outside[ny][nx] = True
                q.append((nx, ny))
    return [[mask[y][x] or not outside[y][x] for x in range(w)] for y in range(h)]


def shape_at(img, x, y, half, seed=None):
    """Outline (a run of fractional points) of the hold at (x, y) (fractions). half is the half-width of the crop
    square (fraction of the photo width). Without a seed, the color around the tapped spot becomes the hold color.
    None when untellable."""
    W, H = img.size
    hx, hy = half * W, half * W
    box = (max(0, int(x * W - hx)), max(0, int(y * H - hy)), min(W, int(x * W + hx) + 1), min(H, int(y * H + hy) + 1))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    crop = img.crop(box)
    scale = WORK / max(crop.size)
    w, h = max(4, int(crop.width * scale)), max(4, int(crop.height * scale))
    px = crop.resize((w, h)).load()
    hsv = [[_hsv(px[i, j]) for i in range(w)] for j in range(h)]
    cx = min(max(int((x * W - box[0]) / crop.width * w), 0), w - 1)
    cy = min(max(int((y * H - box[1]) / crop.height * h), 0), h - 1)
    near = [(hsv[j][i], i, j) for j in range(max(0, cy - 2), min(h, cy + 3)) for i in range(max(0, cx - 2), min(w, cx + 3))]
    if seed is None:
        seed = _mean([c for c, _, _ in near])
    # Start tracing at the pixel nearest the center that looks most like the hold color
    start = next(((i, j) for c, i, j in sorted(near, key=lambda n: abs(n[1] - cx) + abs(n[2] - cy)) if same(c, seed)), None)
    if not start:
        return None
    mask, count = _grow(hsv, w, h, start[0], start[1], seed)
    if count < 6 or count > 0.6 * w * h:
        return None
    border = [mask[j][i] for j in range(h) for i in range(w) if i in (0, w - 1) or j in (0, h - 1)]
    if sum(border) > 0.12 * len(border):  # spills far over the square edge → connects to the wall color
        return None
    mask = _fill_holes(mask, w, h)
    outline = _simplify(_trace(mask, w, h), tol=0.7)
    if len(outline) < 3:
        return None
    if len(outline) > MAX_POINTS:
        step = len(outline) / MAX_POINTS
        outline = [outline[int(k * step)] for k in range(MAX_POINTS)]
    return [[round((box[0] + (i + 0.5) / w * crop.width) / W, 4), round((box[1] + (j + 0.5) / h * crop.height) / H, 4)]
            for i, j in outline]


def refine(img, hold):
    """Points of an AI candidate (fractional {"x", "y", "r", "points"?) with the outline fixed by photo color.
    None when unfixable."""
    img = img.convert("RGB")
    size = hold["r"]
    if hold.get("points"):
        xs, ys = [p[0] for p in hold["points"]], [p[1] for p in hold["points"]]
        size = max(size, (max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2 * img.height / img.width)
    return shape_at(img, hold["x"], hold["y"], max(size * 2.2, 0.03))


def refine_all(img, holds):
    """Fix every candidate outline by color where fixable (keep the AI outline otherwise)."""
    img = img.convert("RGB")
    for hold in holds:
        better = refine(img, hold)
        if better:
            hold["points"] = better
    return holds


def outline_at(img, point):
    """Outline of the hold at the tapped spot (fractions). The hold size is unknown, so try small squares first and
    use the first shape told apart without spilling over the edge. None when untellable."""
    img = img.convert("RGB")
    for half in (0.04, 0.08, 0.14):
        pts = shape_at(img, point["x"], point["y"], half)
        if pts:
            return pts
    return None


def _inside(x, y, pts):
    hit = False
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def _covers(hold, x, y, W, H):
    """Whether (x, y) (fractions) is inside that candidate."""
    if hold.get("points"):
        return _inside(x, y, hold["points"])
    return ((hold["x"] - x) * W) ** 2 + ((hold["y"] - y) * H) ** 2 <= (hold["r"] * W) ** 2


CHAIN_ROUNDS = 6  # rounds of searching around found holds (a problem continues like a chained run of holds)
LINK = 0.25  # blobs within this distance (fraction of the photo's long side) of a found hold join the same problem
PALE_GAP = 0.3
SMALL_PART = 12  # blobs smaller than 1/12 of the tapped hold's pixel count are skipped  # lower bound of the brightness gap from the wall when finding white/black/gray holds


def _wall_color(hsv):
    """Median saturation and value of the most common color (the wall) in the photo."""
    flat = [c for row in hsv[::4] for c in row[::4]]
    s = sorted(c[1] for c in flat)
    v = sorted(c[2] for c in flat)
    return s[len(s) // 2], v[len(v) // 2]


def pale_ok(color, wall):
    """Whether white/black/gray holds may be told apart from the wall (only when brightness differs enough)."""
    return abs(color[2] - wall[1]) >= PALE_GAP


def extend_by_color(img, holds, point, everywhere=False):
    """Find blobs of the tapped hold's color in the photo and add them (pick up what the AI missed).
    Only around found candidates (the bounding box of all candidates, widened). Around added holds too, again
    (problems run on, so trace them like a chain from the tap). everywhere means the whole photo
    (an agent tool. Other walls mix in, but the agent chooses).
    Pale holds like white/black/gray are found by brightness only when far enough from the wall
    (no white on a white wall)."""
    img = img.convert("RGB")
    W, H = img.size
    seed_hold = next((h for h in holds if _covers(h, point["x"], point["y"], W, H)), None)
    if seed_hold is None:  # even when the AI candidate is slightly off the tap, search by the tap's color
        pts = outline_at(img, point)
        if not pts:
            return holds
        seed_hold = {"x": point["x"], "y": point["y"], "r": 0.01, "points": pts,
                     "color": next((h.get("color", "") for h in holds), "")}
        holds = holds + [seed_hold]  # the tapped hold itself also joins the problem holds
    scale = SCAN / max(W, H)
    w, h = max(8, int(W * scale)), max(8, int(H * scale))
    px = img.resize((w, h)).load()
    hsv = [[_hsv(px[x, y]) for x in range(w)] for y in range(h)]
    inside = [hsv[y][x] for y in range(h) for x in range(w) if _covers(seed_hold, (x + 0.5) / w, (y + 0.5) / h, W, H)]
    if len(inside) < 3:
        return holds
    # Skip dots far smaller than the tapped hold (tape scraps, screws, dirt). A real photo picked up red dots all over the wall
    min_blob = max(MIN_BLOB, len(inside) // SMALL_PART)
    # The tapped hold's color (mean of the saturated half inside the outline, so mixed-in wall pixels from a loose outline are fine)
    inside.sort(key=lambda c: c[1], reverse=True)
    color = _mean(inside[:max(3, len(inside) // 2)])
    if not colored(color):
        # White/black/gray: brightness of the mean of the half farthest from the wall
        wall = _wall_color(hsv)
        inside.sort(key=lambda c: abs(c[2] - wall[1]), reverse=True)
        color = _mean(inside[:max(3, len(inside) // 2)])
        if colored(color) or not pale_ok(color, wall):
            return holds
        min_blob = max(MIN_BLOB, len(inside) // 6)  # skip tiny dots of bolt holes and hold shadows
    near = [[same(hsv[y][x], color) for x in range(w)] for y in range(h)]
    seen = [[False] * w for _ in range(h)]
    blobs = []
    for sy in range(h):
        for sx in range(w):
            if not near[sy][sx] or seen[sy][sx]:
                continue
            blob, q = [], deque([(sx, sy)])
            seen[sy][sx] = True
            while q:
                x, y = q.popleft()
                blob.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and near[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        q.append((nx, ny))
            if not (min_blob <= len(blob) <= 0.03 * w * h):  # too-small dots and too-large faces are not holds
                continue
            # Hue is judged loosely per pixel, so compare the whole blob's hue once more (tell red from orange apart)
            if colored(color) and _hue_gap(_mean([hsv[y][x] for x, y in blob])[0], color[0]) >= BLOB_HUE:
                continue
            bx = [p[0] for p in blob]
            by = [p[1] for p in blob]
            cx, cy = sum(bx) / len(bx) / w + 0.5 / w, sum(by) / len(by) / h + 0.5 / h
            r = max(max(bx) - min(bx) + 1, (max(by) - min(by) + 1) * W / H * h / w) / 2 / w
            blobs.append((cx, cy, r))
    extra = []
    for round_ in range(1 if everywhere else CHAIN_ROUNDS):
        xs = [hh["x"] for hh in holds + extra]
        ys = [hh["y"] for hh in holds + extra]
        # Widen the bounding box of all candidates by a quarter of its width/height (see at least 15% of the photo
        # even with few candidates. Widening too far picks up other problems on the next wall)
        # Outside the box, still add blobs near any found hold (problems climb up as a run of holds)
        px_, py_ = max((max(xs) - min(xs)) / 4, 0.15), max((max(ys) - min(ys)) / 4, 0.15)
        area = (0, 0, 1, 1) if everywhere else (min(xs) - px_, min(ys) - py_, max(xs) + px_, max(ys) + py_)
        if round_:  # from the second round, only near found holds (the box widens as holds grow and would pick up the whole wall)
            area = (2, 2, 2, 2)
        side = max(W, H)
        added = 0
        for cx, cy, r in blobs:
            linked = any(math.hypot((cx - hh["x"]) * W, (cy - hh["y"]) * H) <= LINK * side for hh in holds + extra)
            if not (linked or area[0] <= cx <= area[2] and area[1] <= cy <= area[3]):
                continue
            if any(_covers(hh, cx, cy, W, H) for hh in holds + extra):
                continue
            hold = {"x": round(cx, 4), "y": round(cy, 4), "r": round(min(max(r, 0.008), 0.2), 4),
                    "color": seed_hold.get("color", "")}
            better = shape_at(img, cx, cy, max(r * 2.5, 0.03), seed=color)
            if better:
                hold["points"] = better
            extra.append(hold)
            added += 1
            if len(extra) >= MAX_EXTRA:
                return holds + extra
        if not added:
            break
    return holds + extra
