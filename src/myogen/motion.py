"""Intent generator: smooth, natural hand/wrist trajectories from a low-dimensional posture latent.

The latent is a hand-agnostic description of a posture (the same space a decoder can target):

    0..4  closure of thumb, index, middle, ring, little   (0 = extended, 1 = fully flexed)
    5     thumb abduction / opposition                    (0 = adducted, 1 = abducted)
    6     finger spread                                   (0 = together, 1 = spread)
    7     wrist flexion / extension                       (-1 .. 1, sign follows MyoHand flexion_r)
    8     wrist deviation                                 (-1 .. 1, sign follows deviation_r)
    9     pronation / supination                          (-1 .. 1, sign follows pro_sup_r)

Trajectories are minimum-jerk transitions between sampled latent postures, with holds and
returns to rest, so velocities and accelerations are smooth and physiological.
"""
from dataclasses import dataclass

import numpy as np

LATENT_NAMES = ["thumb", "index", "middle", "ring", "little", "thumb_abd", "spread",
                "wrist_flex", "wrist_dev", "pro_sup"]
N_LATENT = len(LATENT_NAMES)

# Relaxed hand (latent).
REST = np.array([0.25, 0.2, 0.25, 0.3, 0.3, 0.35, 0.3, 0.0, 0.0, 0.0])

# Named prototypes; sampled with jitter so the dataset covers functional grasps densely.
PROTOTYPES = {
    "rest":     REST,
    "open":     np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, -0.2, 0.0, 0.0]),
    "power":    np.array([0.8, 0.9, 0.95, 0.95, 0.95, 0.5, 0.2, -0.2, 0.0, 0.0]),
    "fist":     np.array([0.9, 1.0, 1.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
    "tip_pinch": np.array([0.6, 0.6, 0.15, 0.15, 0.15, 0.7, 0.3, 0.0, 0.0, 0.0]),
    "tripod":   np.array([0.6, 0.55, 0.55, 0.2, 0.2, 0.7, 0.3, 0.0, 0.0, 0.0]),
    "key":      np.array([0.5, 0.8, 0.9, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0, 0.3]),
    "point":    np.array([0.7, 0.0, 0.95, 0.95, 0.95, 0.3, 0.0, 0.0, 0.0, 0.0]),
    "thumbs_up": np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0]),
    "spread":   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]),
}

# Correlation of digit closures (enslaving between neighbouring fingers + a global grasp factor).
_DIGIT_CORR = np.array([
    [1.0, 0.5, 0.4, 0.35, 0.3],
    [0.5, 1.0, 0.7, 0.5, 0.4],
    [0.4, 0.7, 1.0, 0.8, 0.6],
    [0.35, 0.5, 0.8, 1.0, 0.85],
    [0.3, 0.4, 0.6, 0.85, 1.0],
])
_DIGIT_CHOL = np.linalg.cholesky(_DIGIT_CORR)


def sample_posture(rng, p_prototype=0.45, wrist_scale=1.0):
    """Sample one latent posture: a jittered prototype or a correlated random posture."""
    if rng.random() < p_prototype:
        z = PROTOTYPES[rng.choice(list(PROTOTYPES))].copy()
        z[:7] += rng.normal(0, 0.08, 7)
    else:
        g = _DIGIT_CHOL @ rng.normal(size=5)
        z = np.empty(N_LATENT)
        z[:5] = 1 / (1 + np.exp(-(1.4 * g + rng.normal(0, 1.0))))  # shared offset = global grasp level
        z[5] = rng.beta(2, 2)
        z[6] = rng.beta(2, 2) * (1 - 0.7 * z[1:5].mean())             # can't spread a closed fist
    # Wrist is (mostly) independent of the hand; often near neutral.
    wrist = rng.normal(0, 0.45, 3) * wrist_scale * (rng.random(3) < 0.6)
    z[7:] = wrist
    lo = np.r_[np.zeros(7), -np.ones(3)]
    return np.clip(z, lo, 1.0)


@dataclass
class JointMap:
    """Latent -> 23 MyoHand joint angles. Per-subject coupling noise makes the mapping non-trivial."""
    dip_ratio: np.ndarray    # (4,) DIP/PIP coupling per finger
    mcp_share: np.ndarray    # (4,) fraction of closure taken by MCP vs PIP
    qmin: np.ndarray
    qmax: np.ndarray

    @classmethod
    def sample(cls, rng, qmin, qmax):
        return cls(dip_ratio=rng.uniform(0.55, 0.9, 4), mcp_share=rng.uniform(0.75, 1.0, 4),
                   qmin=qmin, qmax=qmax)

    def __call__(self, z):
        """z: (..., 10) -> q: (..., 23)."""
        z = np.asarray(z)
        q = np.zeros(z.shape[:-1] + (23,))
        lerp = lambda a, b, t: a + (b - a) * t
        # Wrist: pro_sup, deviation, flexion (signs follow the joint axes; scaled to the joint range).
        q[..., 0] = z[..., 9] * 1.3
        q[..., 1] = np.where(z[..., 8] > 0, z[..., 8] * 0.4, z[..., 8] * 0.17)
        # Tenodesis: stretched finger extensors limit wrist flexion when the hand is closed.
        closure = z[..., 1:5].mean(-1)
        q[..., 2] = z[..., 7] * 0.75 * np.where(z[..., 7] > 0, 1 - 0.55 * closure, 1.0)
        # Thumb (negative = flexion toward the palm for cmc_flexion, mp and ip).
        t, ta = z[..., 0], z[..., 5]
        q[..., 3] = lerp(0.3, -0.6, t)
        q[..., 4] = lerp(-0.4, 0.7, ta)
        q[..., 5] = lerp(0.2, -0.6, t)
        q[..., 6] = lerp(0.2, -1.0, t)
        # Fingers 2..5: mcp_flex, mcp_abd, pip, dip. Spread fans the fingers away from the middle finger.
        fan = np.array([-1.0, -0.2, 0.6, 1.0])
        for k in range(4):
            c = z[..., 1 + k]
            base = 7 + 4 * k
            q[..., base] = 1.45 * np.clip(c * self.mcp_share[k] / 0.9, 0, 1)
            q[..., base + 1] = 0.22 * fan[k] * (z[..., 6] - 0.3) * (1 - c)
            q[..., base + 2] = 1.5 * c
            q[..., base + 3] = self.dip_ratio[k] * 1.5 * c
        return np.clip(q, self.qmin, self.qmax)


# Joint excursion (rad) per unit of latent, and peak angular speed allowed (rad/s): physiological limits that
# keep transitions within what the muscles can produce (fast pro/supination would otherwise saturate the wrist).
_LATENT_SPAN = np.array([1.0, 1.5, 1.5, 1.5, 1.5, 1.1, 0.3, 0.75, 0.4, 1.3])
_LATENT_VMAX = np.array([5.0, 6.0, 6.0, 6.0, 6.0, 4.0, 2.0, 4.0, 3.0, 3.0])


def min_jerk(t):
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


def intent_trajectory(rng, duration, fs, speed=1.0):
    """Latent trajectory (T, 10) sampled at fs: transitions, holds and returns to rest."""
    T = int(round(duration * fs))
    z = np.empty((T, N_LATENT))
    cur = REST + np.r_[rng.normal(0, 0.05, 7), np.zeros(3)]
    i = 0
    while i < T:
        nxt = REST.copy() if rng.random() < 0.2 else sample_posture(rng)
        # Min-jerk peak speed = 1.875 * excursion / duration.
        t_min = 1.875 * np.max(np.abs(nxt - cur) * _LATENT_SPAN / _LATENT_VMAX)
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
    """Slowly varying non-negative scalar (co-contraction, load) with random episodes."""
    x = np.full(T, base)
    i = 0
    while i < T:
        dur = int(rng.uniform(0.5, 3.0) * fs)
        if rng.random() < p_episode:
            x[i:i + dur] = rng.uniform(base, high)
        i += dur
    # first-order smoothing (forward-backward, zero phase)
    a = np.exp(-1 / (tau * fs))
    for rev in (False, True):
        y = x[::-1] if rev else x
        out = np.empty_like(y)
        acc = y[0]
        for k in range(len(y)):
            acc = a * acc + (1 - a) * y[k]
            out[k] = acc
        x = out[::-1] if rev else out
    return x
