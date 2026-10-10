"""Synthetic 16-frame clips with KNOWN dynamics, to measure what a frozen encoder can score when
the world is perfectly well-behaved, and to check that each metric reacts as it should.

Conditions (all 224x224, textured background, one coloured object unless noted):
  static_noise   nothing happens; sensor noise + 1px shake              -> floor of the pixel->encoder pipeline
  linear_clean   constant-velocity translation, no nuisance             -> best case for a frozen encoder
  linear_shake   + random whole-frame translation (+-4px), camera shake
  linear_noise   + sensor noise (sigma 0.03) and +-5% gain flicker
  linear_both    both nuisances (closest to a real handheld clip)
  accel_clean    smooth ease-in translation (non-uniform speed)
  out_and_back   goes out and returns: start ~ end, path not monotone
  step_change    8 objects appear one by one (staircase of discrete 'procedure steps')
  fill_level     container fills at constant rate (continuous state change, 'pouring')
  iid_scenes     a different random scene every frame                   -> iid floor
Progress is monotone and known for linear_*, accel_clean, step_change, fill_level.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift

H = W = 224
T = 16
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
AMPLITUDE_CONDITIONS = [f"linear_r{r}" for r in (6, 12, 24, 48, 90)]  # clean constant-velocity motion, object radius in px
CONDITIONS = ["static_noise", "linear_clean", "linear_shake", "linear_noise", "linear_both", "accel_clean",
              "out_and_back", "step_change", "fill_level", "iid_scenes"]


def _g(rng, sigma):
    g = gaussian_filter(rng.standard_normal((H, W)), sigma)
    return g / g.std()


def texture(rng):
    """Natural-ish background: multi-scale luminance texture, muted per-clip tint, faint chroma noise."""
    lum = rng.uniform(0.35, 0.65) + 0.10 * _g(rng, 20) + 0.06 * _g(rng, 8) + 0.03 * _g(rng, 3)
    tint = rng.uniform(0.85, 1.15, 3)
    chroma = np.stack([0.02 * _g(rng, 12) for _ in range(3)], -1)
    return np.clip(lum[..., None] * tint + chroma, 0.05, 0.95).astype(np.float32)


def disc(img, cx, cy, r, color):
    d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    m = np.clip((r - d) / 1.5 + 0.5, 0, 1)[..., None]
    shade = (0.7 + 0.3 * np.clip(1 - d / r, 0, 1))[..., None]
    return img * (1 - m) + color * shade * m


def rand_color(rng):
    c = rng.uniform(0.0, 1.0, 3)
    return (c / c.max()).astype(np.float32)


def endpoints(rng, margin=40, min_dist=90):
    while True:
        p0, p1 = rng.uniform(margin, H - margin, 2), rng.uniform(margin, H - margin, 2)
        if np.linalg.norm(p1 - p0) >= min_dist:
            return p0, p1


def nuisance(frames, rng, shake, noise):
    out = []
    for f in frames:
        if shake:
            s = rng.integers(-shake, shake + 1, 2) if isinstance(shake, int) else rng.uniform(-shake, shake, 2)
            f = np.stack([ndshift(f[..., c], s, order=1, mode="nearest") for c in range(3)], -1)
        if noise:
            f = f * (1 + 0.05 * rng.standard_normal()) + noise * rng.standard_normal(f.shape).astype(np.float32)
        out.append(np.clip(f, 0, 1))
    return out


def render(cond, rng):
    tau = np.linspace(0, 1, T)
    if cond == "iid_scenes":
        frames = []
        for _ in range(T):
            p0, _ = endpoints(rng)
            frames.append(disc(texture(rng), p0[0], p0[1], rng.uniform(16, 28), rand_color(rng)))
        return [(f * 255).astype(np.uint8) for f in frames]

    bg = texture(rng)
    if cond == "step_change":
        n_obj = 8
        slots = rng.uniform(40, H - 40, (n_obj, 2))
        cols = [rand_color(rng) for _ in range(n_obj)]
        r = rng.uniform(14, 20)
        n_vis = np.round(n_obj * tau).astype(int)
        frames = []
        for k in n_vis:
            f = bg.copy()
            for j in range(k):
                f = disc(f, slots[j, 0], slots[j, 1], r, cols[j])
            frames.append(f)
    elif cond == "fill_level":
        cw, chh = rng.uniform(55, 75), rng.uniform(90, 120)
        x0, y0 = rng.uniform(40, H - 40 - cw), rng.uniform(30, H - 30 - chh)
        col = rand_color(rng)
        frames = []
        for a in tau:
            f = bg.copy()
            top = y0 + chh * (1 - a)
            inside = (XX >= x0) & (XX <= x0 + cw)
            f[inside & (YY >= y0) & (YY <= y0 + chh)] *= 0.55  # container interior darker
            f[inside & (YY >= top) & (YY <= y0 + chh)] = col
            edge = ((np.abs(XX - x0) < 2) | (np.abs(XX - (x0 + cw)) < 2)) & (YY >= y0) & (YY <= y0 + chh)
            edge |= ((np.abs(YY - (y0 + chh)) < 2) & (XX >= x0) & (XX <= x0 + cw))
            f[edge] = 0.1
            frames.append(f)
    else:
        p0, p1 = endpoints(rng)
        r, col = rng.uniform(16, 28), rand_color(rng)
        if cond.startswith("linear_r"):
            r = float(cond[len("linear_r"):])
            cond = "linear_clean"
        if cond == "static_noise":
            path = np.repeat(p0[None], T, 0)
        elif cond == "accel_clean":
            path = p0 + (tau ** 2)[:, None] * (p1 - p0)
        elif cond == "out_and_back":
            path = p0 + (4 * tau * (1 - tau))[:, None] * (p1 - p0)
        else:
            path = p0 + tau[:, None] * (p1 - p0)
        frames = [disc(bg, p[0], p[1], r, col) for p in path]

    shake = {"static_noise": 1, "linear_shake": 4, "linear_both": 4}.get(cond, 0)
    noise = {"static_noise": 0.03, "linear_noise": 0.03, "linear_both": 0.03}.get(cond, 0.0)
    frames = nuisance(frames, rng, shake, noise)
    return [(f * 255).astype(np.uint8) for f in frames]


def make_dataset(cond, n, seed):
    rng = np.random.default_rng(seed)
    return [render(cond, rng) for _ in range(n)]
