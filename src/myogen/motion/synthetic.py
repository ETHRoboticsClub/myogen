"""Synthetic natural movement: minimum-jerk transitions between sampled postures, with holds and returns to rest."""
import numpy as np

from .latent import LATENT_HI, LATENT_LO, N_LATENT, REST

PROTOTYPES = {
    "rest":      REST,
    "open":      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, -0.2, 0.0, 0.0]),
    "power":     np.array([0.8, 0.9, 0.95, 0.95, 0.95, 0.5, 0.2, -0.2, 0.0, 0.0]),
    "fist":      np.array([0.9, 1.0, 1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
    "tip_pinch": np.array([0.6, 0.6, 0.15, 0.15, 0.15, 0.7, 0.3, 0.0, 0.0, 0.0]),
    "tripod":    np.array([0.6, 0.55, 0.55, 0.2, 0.2, 0.7, 0.3, 0.0, 0.0, 0.0]),
    "key":       np.array([0.5, 0.8, 0.9, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0, 0.3]),
    "point":     np.array([0.7, 0.0, 0.95, 0.95, 0.95, 0.3, 0.0, 0.0, 0.0, 0.0]),
    "thumbs_up": np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0]),
    "spread":    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]),
}

# Correlated digit closures: enslaving between neighbouring fingers.
_DIGIT_CHOL = np.linalg.cholesky(np.array([
    [1.0, 0.5, 0.4, 0.35, 0.3],
    [0.5, 1.0, 0.7, 0.5, 0.4],
    [0.4, 0.7, 1.0, 0.8, 0.6],
    [0.35, 0.5, 0.8, 1.0, 0.85],
    [0.3, 0.4, 0.6, 0.85, 1.0],
]))

# Joint excursion (rad) per unit of latent and peak angular speed (rad/s): keeps transitions physiological.
_LATENT_SPAN = np.array([1.0, 1.5, 1.5, 1.5, 1.5, 1.1, 0.3, 0.75, 0.4, 1.3])
_LATENT_VMAX = np.array([5.0, 6.0, 6.0, 6.0, 6.0, 4.0, 2.0, 4.0, 3.0, 3.0])


def sample_posture(rng, p_prototype=0.45, wrist_scale=1.0):
    """One latent posture: a jittered prototype or a random posture with correlated digits."""
    if rng.random() < p_prototype:
        z = PROTOTYPES[rng.choice(list(PROTOTYPES))].copy()
        z[:7] += rng.normal(0, 0.08, 7)
    else:
        g = _DIGIT_CHOL @ rng.normal(size=5)
        z = np.empty(N_LATENT)
        z[:5] = 1 / (1 + np.exp(-(1.4 * g + rng.normal(0, 1.0))))   # shared offset = global grasp level
        z[5] = rng.beta(2, 2)
        z[6] = rng.beta(2, 2) * (1 - 0.7 * z[1:5].mean())              # a closed fist cannot spread
    z[7:] = rng.normal(0, 0.45, 3) * wrist_scale * (rng.random(3) < 0.6)   # wrist: independent, often neutral
    return np.clip(z, LATENT_LO, LATENT_HI)


def min_jerk(t):
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def intent_trajectory(rng, duration, fs, speed=1.0):
    """(T, 10) latent trajectory at fs: transitions, holds and returns to rest."""
    T = int(round(duration * fs))
    z = np.empty((T, N_LATENT))
    cur = REST + np.r_[rng.normal(0, 0.05, 7), np.zeros(3)]
    i = 0
    while i < T:
        nxt = REST.copy() if rng.random() < 0.2 else sample_posture(rng)
        t_min = 1.875 * np.max(np.abs(nxt - cur) * _LATENT_SPAN / _LATENT_VMAX)   # min-jerk peak speed limit
        move = max(1, int(max(rng.uniform(0.35, 1.4) / speed, t_min) * fs))
        hold = int(rng.exponential(0.6) * fs) if rng.random() < 0.7 else 0
        s = min_jerk(np.linspace(0, 1, move, endpoint=False))[:, None]
        seg = np.vstack([cur + (nxt - cur) * s, np.repeat(nxt[None], hold, 0)])
        n = min(len(seg), T - i)
        z[i:i + n] = seg[:n]
        i += n
        cur = nxt
    return z


def smooth_profile(rng, T, fs, base, high, p_episode, tau=0.3):
    """(T,) slowly varying scalar at `base` with random episodes up to `high`, zero-phase smoothed with tau."""
    x = np.full(T, base)
    i = 0
    while i < T:
        dur = int(rng.uniform(0.5, 3.0) * fs)
        if rng.random() < p_episode:
            x[i:i + dur] = rng.uniform(base, high)
        i += dur
    a = np.exp(-1 / (tau * fs))
    return _one_pole(_one_pole(x, a)[::-1], a)[::-1]       # forward, then backward: zero phase


def _one_pole(x, a):
    """First-order low-pass y[k] = a y[k-1] + (1 - a) x[k], starting at x[0]."""
    out = np.empty_like(x)
    acc = x[0]
    for k in range(len(x)):
        acc = a * acc + (1 - a) * x[k]
        out[k] = acc
    return out
