"""The 10-D posture latent (a hand-agnostic description of a hand/wrist posture) and its mapping to joints.

    column  name        range    meaning
    0..4    thumb, index, middle, ring, little
                        0 .. 1   closure (0 = extended, 1 = fully flexed)
    5       thumb_abd   0 .. 1   thumb abduction / opposition (0 = adducted)
    6       spread      0 .. 1   finger spread (0 = together)
    7       wrist_flex -1 .. 1   wrist flexion (+) / extension (-)
    8       wrist_dev  -1 .. 1   wrist deviation (sign of deviation_r)
    9       pro_sup    -1 .. 1   pronation / supination (sign of pro_sup_r)
"""
from dataclasses import dataclass

import numpy as np

LATENT_NAMES = ["thumb", "index", "middle", "ring", "little", "thumb_abd", "spread",
                "wrist_flex", "wrist_dev", "pro_sup"]
N_LATENT = len(LATENT_NAMES)
LATENT_LO = np.r_[np.zeros(7), -np.ones(3)]
LATENT_HI = np.ones(N_LATENT)

REST = np.array([0.25, 0.2, 0.25, 0.3, 0.3, 0.35, 0.3, 0.0, 0.0, 0.0])    # relaxed hand


@dataclass
class JointMap:
    """Latent -> the 23 hand/wrist joint angles (model.HAND_JOINTS), with per-subject finger coupling."""
    dip_ratio: np.ndarray    # (4,) DIP / PIP coupling per finger
    mcp_share: np.ndarray    # (4,) fraction of closure taken by the MCP vs the PIP
    qmin: np.ndarray
    qmax: np.ndarray

    @classmethod
    def sample(cls, rng, qmin, qmax):
        return cls(dip_ratio=rng.uniform(0.55, 0.9, 4), mcp_share=rng.uniform(0.75, 1.0, 4), qmin=qmin, qmax=qmax)

    def __call__(self, z):
        """z: (..., 10) -> q: (..., 23)."""
        z = np.asarray(z)
        q = np.zeros(z.shape[:-1] + (23,))
        lerp = lambda a, b, t: a + (b - a) * t
        # Wrist: pro_sup, deviation, flexion.
        q[..., 0] = z[..., 9] * 1.3
        q[..., 1] = np.where(z[..., 8] > 0, z[..., 8] * 0.4, z[..., 8] * 0.17)
        closure = z[..., 1:5].mean(-1)
        # Tenodesis: stretched finger extensors limit wrist flexion when the hand is closed.
        q[..., 2] = z[..., 7] * 0.75 * np.where(z[..., 7] > 0, 1 - 0.55 * closure, 1.0)
        # Thumb (negative = flexion toward the palm).
        t, ta = z[..., 0], z[..., 5]
        q[..., 3] = lerp(0.3, -0.6, t)
        q[..., 4] = lerp(-0.4, 0.7, ta)
        q[..., 5] = lerp(0.2, -0.6, t)
        q[..., 6] = lerp(0.2, -1.0, t)
        # Fingers 2..5: MCP flexion, MCP abduction, PIP, DIP. Spread fans the fingers away from the middle one.
        fan = np.array([-1.0, -0.2, 0.6, 1.0])
        for k in range(4):
            c = z[..., 1 + k]
            base = 7 + 4 * k
            q[..., base] = 1.45 * np.clip(c * self.mcp_share[k] / 0.9, 0, 1)
            q[..., base + 1] = 0.22 * fan[k] * (z[..., 6] - 0.3) * (1 - c)
            q[..., base + 2] = 1.5 * c
            q[..., base + 3] = self.dip_ratio[k] * 1.5 * c
        return np.clip(q, self.qmin, self.qmax)
