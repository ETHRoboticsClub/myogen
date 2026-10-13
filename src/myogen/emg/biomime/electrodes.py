"""Electrode layouts on BioMime's 10 x 32 monopolar grid.

A channel is a weighted sum of poles at (angle, z): angle as a fraction of the circumference (column = 32 * angle),
z as a fraction of forearm length (rows via GridGeometry). Bipolar = (+1, -1) along the fibres; positions are
interpolated bilinearly, poles beyond the grid's rows are clamped and counted.

BioMime has no fat condition: fat thicker than its reference forearm is an extra spatial low-pass + attenuation.
"""
from dataclasses import dataclass

import numpy as np

from .anatomy import FAT_REF, GridGeometry
from .generator import N_COL, N_ROW

FAT_ATTEN_DECAY = 0.006   # m, same e-folding as BioMime's depth decay


@dataclass
class Channel:
    poles: list            # [(angle, z_frac, weight), ...]
    name: str = ""


def ring_channels(n, angle0, z_frac, ied_frac, stagger=0.0, jitter=None, prefix="ring"):
    """n bipolar channels equally spaced around the forearm, poles along the fibres (z +- ied/2)."""
    jitter = np.zeros(n) if jitter is None else jitter
    chans = []
    for k in range(n):
        a = angle0 + stagger + k / n + jitter[k]
        chans.append(Channel([(a, z_frac - ied_frac / 2, 1.0), (a, z_frac + ied_frac / 2, -1.0)], f"{prefix}{k}"))
    return chans


def interp_matrix(channels, grid: GridGeometry):
    """(C, 320) weights on the flattened (row, col) grid; also returns how many poles fell outside the rows."""
    W = np.zeros((len(channels), N_ROW * N_COL))
    outside = 0
    for c, ch in enumerate(channels):
        for a, z, w in ch.poles:
            col = (a % 1.0) * N_COL
            row = float(grid.row_of(z))
            if row < 0 or row > N_ROW - 1:
                outside += 1
            row = np.clip(row, 0, N_ROW - 1)
            r0, c0 = int(np.floor(row)), int(np.floor(col))
            fr, fc = row - r0, col - c0
            for dr, wr in ((0, 1 - fr), (1, fr)):
                rr = min(r0 + dr, N_ROW - 1)
                for dc, wc in ((0, 1 - fc), (1, fc)):
                    W[c, rr * N_COL + (c0 + dc) % N_COL] += w * wr * wc
    return W, outside


def skin_filter(grid: GridGeometry, fat):
    """(320, 320) spatial low-pass x attenuation for fat thicker than BioMime's reference forearm."""
    extra = max(0.0, fat - FAT_REF)
    if extra <= 0:
        return np.eye(N_ROW * N_COL)
    sigma = extra                                            # m on the skin surface
    rows = np.arange(N_ROW) * grid.row_pitch
    cols = np.arange(N_COL)
    dr = rows[:, None] - rows[None, :]
    dcol = np.minimum(np.abs(cols[:, None] - cols[None, :]), N_COL - np.abs(cols[:, None] - cols[None, :]))
    Kr = np.exp(-0.5 * (dr / sigma) ** 2)
    Kc = np.exp(-0.5 * (dcol * grid.col_pitch / sigma) ** 2)
    Kr /= Kr.sum(1, keepdims=True)
    Kc /= Kc.sum(1, keepdims=True)
    return np.kron(Kr, Kc) * np.exp(-extra / FAT_ATTEN_DECAY)


def subject_channels(layout, session, forearm_length, territory_angle=None, ied=0.02, rng=None):
    """Channels for one of our ElectrodeLayouts (rings, staggered rings, targeted) in BioMime coordinates.

    territory_angle: muscle -> (angle_lo, angle_hi), used to place targeted electrodes over a muscle belly.
    """
    L = forearm_length
    ied_frac = ied / L
    per_ring = layout.per_ring
    rng = rng or np.random.default_rng(0)
    angle0 = (layout.rotation + session.rotation_shift) / (2 * np.pi)
    chans = []
    for r in range(layout.rings):
        z = layout.level + session.level_shift + r * layout.ring_gap / L
        jitter = rng.normal(0, layout.angle_jitter / (2 * np.pi), per_ring)
        chans += ring_channels(per_ring, angle0, z, ied_frac, stagger=0.5 / per_ring * r, jitter=jitter,
                               prefix=f"ring{r}_")
    for name in layout.targets:
        lo, hi = territory_angle[name]
        a = 0.5 * (lo + hi) + 0.3 * session.rotation_shift / (2 * np.pi)
        z = layout.target_level + session.level_shift
        chans.append(Channel([(a, z - ied_frac / 2, 1.0), (a, z + ied_frac / 2, -1.0)], f"target_{name}"))
    return chans
