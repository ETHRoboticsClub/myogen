"""Map MuJoCo muscles into BioMime's forearm: motor-unit territories, counts and sizes, grid geometry.

BioMime encodes one numerically modelled forearm. Motor-unit position is given by (depth below the skin, angle
as a fraction of the circumference); the 10 x 32 grid has 32 columns around the forearm (column = 32 * angle)
and 10 rows along the fibres.

Territories:
  * anchors: the 8 muscle compartments BioMime was trained on, with the territory ranges published in NeuroMotion
    (NeuroMotion/MNPoollib/mn_params.py, Ma et al. 2024). These match the volume conductor exactly.
  * other muscles: our MuJoCo line-of-action geometry mapped into BioMime angles with a rotation fitted on the
    anchors (~33 deg RMS residual); depth = skin radius - line-of-action radius scaled into the reference
    cross-section (same mapping as the analytic EMG model); extent from the muscle size.
    Depths beyond the training range (deep flexors, supinator) are clipped and compensated with an attenuation.
  * ANATOMY_OVERRIDES: muscles whose MuJoCo line of action is misleading at the electrode level (it passes near
    the radius), placed from standard cross-sectional anatomy instead.

Grid axial placement is NOT documented by BioMime/NeuroMotion; GridGeometry makes the assumption explicit:
rows at the standard 10 mm HD-grid pitch starting ~1.5 cm below the elbow (6-42 % of forearm length, proximal
first), i.e. covering the proximal bellies where armbands sit. It can be recalibrated without regenerating
MUAPs (`myogen biomime-bank regrid`).
"""
from dataclasses import dataclass, field

import numpy as np

from ..model import belly_points
from .model import RANGES

# NeuroMotion mn_params.py (Ma et al. 2024): depth (m), angle (fraction), area (mm^2), number of MUs.
ANCHOR_TABLE = {
    "ECRB": ([0.0130, 0.0220], [0.4946, 0.6632], 60.100, 186),
    "ECRL": ([0.0085, 0.0153], [0.4607, 0.5109], 70.715, 204),
    "PL": ([0.0071, 0.0114], [0.0540, 0.0956], 35.335, 164),
    "FCU": ([0.0074, 0.0168], [0.7878, 1.0000], 162.23, 422),
    "ECU": ([0.0092, 0.0168], [0.7194, 0.7779], 47.720, 180),
    "ED": ([0.0079, 0.0171], [0.5637, 0.6826], 60.000, 186),
    "FDS": ([0.0169, 0.0231], [0.1471, 0.2264], 25.335, 158),
}
# Our muscles -> anchor compartment. Finger compartments split the anchor's angular range (index radial-most:
# radial side = lower angle for the extensors, higher angle for the flexors in BioMime's convention).
ANCHOR_OF = {"ECRB_r": "ECRB", "ECRL_r": "ECRL", "PL_r": "PL", "FCU_r": "FCU", "ECU_r": "ECU"}
FINGER_SPLIT = {"EDC": ("ED", ["EDC2_r", "EDC3_r", "EDC4_r", "EDC5_r"], +1),
                "FDS": ("FDS", ["FDS2_r", "FDS3_r", "FDS4_r", "FDS5_r"], -1)}
# Bellies above the elbow (not under forearm electrodes).
UPPER_ARM = ("BIClong_r", "BICshort_r", "BRA_r", "TRIlong_r", "TRIlat_r", "TRImed_r")

# (angle range, depth range in m) from cross-sectional anatomy, in BioMime's angle convention (PL ~0.07 palmar
# midline, FDS ~0.19, ECRL ~0.49, ECU ~0.75, FCU ~0.9).
ANATOMY_OVERRIDES = {
    "BRD_r": ((0.33, 0.45), (0.007, 0.014)),   # superficial, radial border between the flexors and ECRL
    "FPL_r": ((0.20, 0.30), (0.024, 0.030)),   # deep, palmar-radial, under FDS
}
R_MODEL_REF = 0.032          # m, outermost MyoHand/MyoArm line-of-action radius (as in src/emg.py)
FAT_REF = 0.003              # m, assumed subcutaneous fat of BioMime's reference forearm
RADIAL_FILL = 0.9

# Fibres per muscle from PCSA (max isometric force / 50.8 N/cm^2) and NeuroMotion's cadaver-corrected fibre
# density (34 um fibres, FDI correction -> ~19.8k fibres per cm^2 of PCSA, i.e. 40k-140k per forearm muscle).
# The simpler 200 fibres/mm^2 x territory area in NeuroMotion's example script gives ~8x fewer fibres and
# ~20x too little sEMG amplitude.
SPECIFIC_TENSION = 50.8      # N / cm^2
FIBRES_PER_CM2 = 19_800
MU_PER_MM2 = 3.2             # median of the anchors (number of MUs / area)
DEPTH_EXTRA_DECAY = 0.006    # m: e-folding of BioMime's own depth decay (13x over 7-23 mm), continued beyond its range


@dataclass
class GridGeometry:
    z0_frac: float = 0.06            # forearm-length fraction (from the elbow) of grid row 0
    z1_frac: float = 0.42            # ... of grid row 9 (-> ~10 mm row pitch on a 26 cm forearm)
    circumference_ref: float = 0.25  # m, BioMime forearm circumference (assumed) -> column pitch ~7.8 mm
    forearm_length_ref: float = 0.258  # m, our model's forearm (elbow-lunate)

    def row_of(self, z_frac):
        return (np.asarray(z_frac) - self.z0_frac) / (self.z1_frac - self.z0_frac) * 9.0

    @property
    def row_pitch(self):
        return (self.z1_frac - self.z0_frac) * self.forearm_length_ref / 9.0

    @property
    def col_pitch(self):
        return self.circumference_ref / 32.0


@dataclass
class Territory:
    muscle: str
    depth: tuple                     # m, clipped to BioMime's range
    angle: tuple                     # fraction of circumference (may wrap past 1)
    area_mm2: float
    n_mu: int
    n_fibres: int
    anchored: bool
    depth_excess: float = 0.0        # m beyond BioMime's deepest training depth (-> extra attenuation)
    axial_overlap: float = 1.0       # fraction of the belly within the grid's axial span
    z_range: tuple = field(default=(0.0, 1.0))  # belly extent (forearm-length fraction)

    @property
    def gain(self):
        return float(np.exp(-self.depth_excess / DEPTH_EXTRA_DECAY) * self.axial_overlap)


def _belly_summary(m, d, anat, name, z_ref=0.3):
    P = belly_points(m, d, anat, 24)[anat.muscle_names.index(name)]
    L = anat.forearm_length
    z = P[:, 2] / L
    i = np.argmin(np.abs(z - np.clip(z_ref, z.min(), z.max())))
    return np.arctan2(P[i, 1], P[i, 0]), np.hypot(P[i, 0], P[i, 1]), (float(z.min()), float(z.max())), z


def fit_angle_transform(m, d, anat):
    """Rotation (sign +1) from our forearm-frame angles to BioMime angle fractions, fitted on the anchors."""
    ours, target = [], []
    for name, key in ANCHOR_OF.items():
        ours.append(_belly_summary(m, d, anat, name)[0])
        target.append(np.mean(ANCHOR_TABLE[key][1]) * 2 * np.pi)
    for prefix, (key, names, _) in FINGER_SPLIT.items():
        ours.append(np.angle(np.mean([np.exp(1j * _belly_summary(m, d, anat, n)[0]) for n in names])))
        target.append(np.mean(ANCHOR_TABLE[key][1]) * 2 * np.pi)
    ours, target = np.array(ours), np.array(target)
    offset = np.angle(np.mean(np.exp(1j * (target - ours))))
    resid = np.angle(np.exp(1j * (target - ours - offset)))
    return offset, float(np.degrees(np.sqrt((resid ** 2).mean())))


def build_territories(m, d, anat, grid: GridGeometry = None):
    """Territory for every muscle visible to forearm electrodes, keyed by muscle name."""
    grid = grid or GridGeometry()
    offset, _ = fit_angle_transform(m, d, anat)
    R_skin = grid.circumference_ref / (2 * np.pi)
    area_per_newton = np.median([ANCHOR_TABLE[k][2] / m.actuator_gainprm[anat.muscle_names.index(n), 2]
                                 for n, k in ANCHOR_OF.items()])
    split_of = {n: (key, i, len(names), sgn) for key_, (key, names, sgn) in FINGER_SPLIT.items()
                for i, n in enumerate(names)}
    dmin, dmax = RANGES["depth"]
    terr = {}
    for i, name in enumerate(anat.muscle_names):
        if not anat.extrinsic[i] or name in UPPER_ARM:
            continue
        th, r, zr, zs = _belly_summary(m, d, anat, name)
        overlap = float(np.mean((zs >= grid.z0_frac - 0.05) & (zs <= grid.z1_frac + 0.05)))
        if overlap <= 0.0:
            continue
        F0 = m.actuator_gainprm[i, 2]
        if name in ANCHOR_OF:
            dep, ang, area, n_mu = ANCHOR_TABLE[ANCHOR_OF[name]]
            anchored = True
        elif name in split_of:
            key, k, n, sgn = split_of[name]
            dep, (a0, a1), area, n_mu = ANCHOR_TABLE[key]
            w = (a1 - a0) / n
            j = k if sgn > 0 else n - 1 - k
            ang = [a0 + j * w, a0 + (j + 1) * w]
            area, n_mu = area / 2, max(40, n_mu // 2)       # each finger compartment ~ half of the anchor
            anchored = True
        else:
            area = area_per_newton * F0
            n_mu = int(np.clip(round(MU_PER_MM2 * area), 40, 400))
            if name in ANATOMY_OVERRIDES:
                ang, dep = (list(v) for v in ANATOMY_OVERRIDES[name])
            else:
                centre = ((th + offset) / (2 * np.pi)) % 1.0
                half = 0.015 + 0.0005 * area
                ang = [centre - half, centre + half]
                r_anat = min(r / R_MODEL_REF, 1.0) * (R_skin - FAT_REF) * RADIAL_FILL
                dc = R_skin - r_anat
                hd = 0.002 + 0.00004 * area
                dep = [dc - hd, dc + hd]
            anchored = False
        excess = max(0.0, float(np.mean(dep)) - dmax)
        dep = (float(np.clip(dep[0], dmin, dmax)), float(np.clip(dep[1], dmin, dmax)))
        terr[name] = Territory(
            muscle=name, depth=dep, angle=(float(ang[0]), float(ang[1])), area_mm2=float(area),
            n_mu=int(n_mu), n_fibres=int(F0 / SPECIFIC_TENSION * FIBRES_PER_CM2),
            anchored=anchored,
            depth_excess=excess, axial_overlap=overlap, z_range=zr,
        )
    return terr


def mu_sizes(n_mu, n_fibres):
    """Fibres per motor unit (NeuroMotion Eq. 1-2): linear-exponential twitch distribution, sums to n_fibres."""
    j = np.arange(1, n_mu + 1)
    tw = 0.81 * (18.51 * j / n_mu + 104.10 ** ((j / n_mu) ** 4.83))
    return tw / tw.sum() * n_fibres
