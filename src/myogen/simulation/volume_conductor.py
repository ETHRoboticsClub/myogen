from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import special
from scipy.ndimage import gaussian_filter

from ..params import VolumeConductorParams

log = logging.getLogger(__name__)
TENDON_FIBRES = 9          # fibre lengths averaged over the tendon spread
CACHE_DIR = Path(os.environ.get("MYOGEN_CACHE", Path.home() / ".cache" / "myogen"))


class LayeredCylinder:
    """Potential on the skin of a point current source in a layered cylinder: bone, anisotropic muscle (radial and
    axial conductivity), fat, skin, insulated outside. In the Fourier domain (axial wavenumber k, angular order m)
    every layer's potential is a sum of modified Bessel functions; continuity of the potential and of the normal
    current at each interface, no current through the skin, give 7 equations per (k, m). Coefficients are scaled
    to each layer's interface (Bessel ratios <= 1), so the system stays finite at large k * radius.
    Lengths in metres, conductivities in S/m; the result is volts per ampere."""

    def __init__(self, p: VolumeConductorParams, skin_radius_m: float, fat_m: float) -> None:
        self.p = p
        self.d = skin_radius_m
        self.c = self.d - p.skin_mm / 1000
        self.b = self.c - fat_m
        self.a = p.bone_radius_mm / 1000
        if not 0 < self.a < self.b:
            raise ValueError(f"bone radius {self.a * 1000:.1f} mm must be inside the muscle ({self.b * 1000:.1f} mm)")

    def transfer(self, R: float, k: np.ndarray, m: np.ndarray) -> np.ndarray:
        """(len k, len m) skin potential per unit current of a source at radius R (muscle), k > 0, m >= 0."""
        p = self.p
        s_r, s_z, s_b, s_f, s_s = p.sigma_muscle_radial, p.sigma_muscle_axial, p.sigma_bone, p.sigma_fat, p.sigma_skin
        K, M = np.meshgrid(k, m, indexing="ij")
        ka = K * np.sqrt(s_z / s_r)
        a, b, c, d = self.a, self.b, self.c, self.d
        # Particular solution in the muscle (infinite anisotropic medium) at its two interfaces, and its slope.
        pa = _ik(M, ka * a, ka * R) / s_r
        pb = _ik(M, ka * R, ka * b) / s_r
        dpa, dpb = pa * ka * _li(M, ka * a), pb * ka * _lk(M, ka * b)
        rI_m, rK_m = _ri(M, ka * a, ka * b), _rk(M, ka * b, ka * a)
        rI_f, rK_f = _ri(M, K * b, K * c), _rk(M, K * c, K * b)
        rI_s, rK_s = _ri(M, K * c, K * d), _rk(M, K * d, K * c)
        n = K.shape
        A = np.zeros(n + (7, 7))
        y = np.zeros(n + (7,))
        # unknowns: bone A, muscle B (I) C (K), fat D (I) E (K), skin F (I) G (K)
        A[..., 0, :3] = np.stack([np.ones(n), -rI_m, -np.ones(n)], -1)
        y[..., 0] = pa
        A[..., 1, 0] = s_b * K * _li(M, K * a)
        A[..., 1, 1] = -s_r * ka * rI_m * _li(M, ka * a)
        A[..., 1, 2] = -s_r * ka * _lk(M, ka * a)
        y[..., 1] = s_r * dpa
        A[..., 2, 1:5] = np.stack([np.ones(n), rK_m, -rI_f, -np.ones(n)], -1)
        y[..., 2] = -pb
        A[..., 3, 1] = s_r * ka * _li(M, ka * b)
        A[..., 3, 2] = s_r * ka * rK_m * _lk(M, ka * b)
        A[..., 3, 3] = -s_f * K * rI_f * _li(M, K * b)
        A[..., 3, 4] = -s_f * K * _lk(M, K * b)
        y[..., 3] = -s_r * dpb
        A[..., 4, 3:7] = np.stack([np.ones(n), rK_f, -rI_s, -np.ones(n)], -1)
        A[..., 5, 3] = s_f * _li(M, K * c)
        A[..., 5, 4] = s_f * rK_f * _lk(M, K * c)
        A[..., 5, 5] = -s_s * rI_s * _li(M, K * c)
        A[..., 5, 6] = -s_s * _lk(M, K * c)
        A[..., 6, 5] = _li(M, K * d)
        A[..., 6, 6] = rK_s * _lk(M, K * d)
        x = np.linalg.solve(A, y[..., None])[..., 0]
        return x[..., 5] + x[..., 6] * rK_s

    def point_potential(self, R: float, theta: np.ndarray, n_z: int, dz: float, electrode_radius_m: float = 0.0
                        ) -> np.ndarray:
        """(len theta, n_z) skin potential (V/A) at angles theta from a source at radius R, z on a periodic grid of
        n_z steps of dz (index 0: the source's height). A round electrode of this radius averages it."""
        k = 2 * np.pi * np.fft.rfftfreq(n_z, dz)
        m = np.arange(self.p.angular_orders + 1)
        H = np.zeros((len(k), len(m)))
        H[1:] = self.transfer(R, k[1:], m)             # k = 0: no net source along a fibre, so it never contributes
        if electrode_radius_m > 0:
            kappa = np.sqrt(k[:, None] ** 2 + (m[None] / self.d) ** 2) * electrode_radius_m
            H *= np.where(kappa > 1e-9, 2 * special.j1(kappa) / np.maximum(kappa, 1e-9), 1.0)
        weight = np.where(m == 0, 1.0, 2.0)            # m and -m together (H is even in m)
        angular = np.cos(np.outer(theta, m)) * weight  # (theta, m)
        h = np.fft.irfft(H @ angular.T, n=n_z, axis=0) / (2 * np.pi * dz)   # (z, theta)
        return h.T


class MUAPTable:
    """Motor-unit action potentials at a monopolar skin pole, by the unit's depth below the muscle surface, its
    angle from the pole and the pole's distance along the forearm from the unit's end-plate. Each fibre's
    intracellular action potential (Rosenfalck) starts at the end-plate, propagates both ways at the conduction
    velocity and stops at the tendons (the sources at the end-plate and the fibre ends included); the unit is a
    bundle of fibres around its centre with scattered end-plates (a blur of the single-fibre table).

    Stored compressed: a few waveforms (principal components, `basis`) and their weights over the grid, so the
    engine adds weights into per-waveform impulse trains and convolves each once, as with the width bands. Built
    once per forearm geometry and cached on disk."""

    def __init__(self, p: VolumeConductorParams, skin_radius_m: float, fat_m: float, fs: float) -> None:
        self.p, self.fs = p, fs
        key = {**asdict(p), "skin_radius_mm": round(skin_radius_m * 1000, 1), "fat_mm": round(fat_m * 1000, 1),
               "fs": fs, "version": 2}
        file = CACHE_DIR / f"muap_{hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]}.npz"
        if file.exists():
            data = np.load(file)
        else:
            log.info("building the MUAP table (once per forearm geometry)")
            data = self._build(LayeredCylinder(p, skin_radius_m, fat_m))
            file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file, **data)
        self.depth, self.theta, self.dz = data["depth"], data["theta"], data["dz"]
        self.basis, self.weights = data["basis"], data["weights"]
        self.muscle_radius = float(data["muscle_radius"])

    def _build(self, cyl: LayeredCylinder) -> dict[str, np.ndarray]:
        p = self.p
        dz = p.grid_mm / 1000
        n_z = int(round(p.window_mm / p.grid_mm))
        depth = np.asarray(p.depths_mm) / 1000
        depth = depth[depth < cyl.b - cyl.a - 0.5e-3]
        theta = np.radians(np.arange(0, 180 + 1e-9, p.angle_step_deg))
        dz_axis = np.arange(-p.max_distance_mm, p.max_distance_mm + 1e-9, p.distance_step_mm) / 1000
        t = np.arange(-p.half_window_ms, p.half_window_ms + 1e-9, 1000 / self.fs) / 1000
        z = (np.arange(n_z) - n_z // 2) * dz                       # fibre axis, end-plate at 0
        source = _fibre_current(z, t, p)                           # (t, z) A/m
        S = np.fft.rfft(np.fft.ifftshift(source, axes=1), axis=1)
        pick = np.round(dz_axis / dz).astype(int) % n_z
        table = np.zeros((len(depth), len(theta), len(dz_axis), len(t)))
        for i, dep in enumerate(depth):
            h = cyl.point_potential(cyl.b - dep, theta, n_z, dz, p.electrode_radius_mm / 1000)   # (theta, z)
            Hh = np.fft.rfft(h, axis=1)
            for j in range(len(theta)):
                phi = np.fft.irfft(S * Hh[j][None], n=n_z, axis=1) * dz    # (t, z_electrode)
                table[i, j] = phi[:, pick].T
        # The unit: fibres around its centre and end-plates scattered along the fibre (blurs, in grid steps).
        arc_step = np.radians(p.angle_step_deg) * cyl.b
        sig = (0, p.territory_mm / 1000 / arc_step, p.endplate_spread_mm / p.distance_step_mm, 0)
        table = gaussian_filter(table, sig, mode="nearest")
        depth_step = np.gradient(depth)
        for i in range(len(depth)):                              # depth blur on the non-uniform depth grid
            w = np.exp(-0.5 * ((depth - depth[i]) / (p.territory_mm / 1000)) ** 2) * depth_step
            table[i] = np.tensordot(w / w.sum(), table, axes=(0, 0))
        flat = table.reshape(-1, len(t))
        norm = np.linalg.norm(flat, axis=1)
        shapes = flat[norm > 0] / norm[norm > 0, None]          # every waveform counts, however weak
        basis = np.linalg.svd(shapes[:: max(1, len(shapes) // 200000)], full_matrices=False)[2][: p.n_basis]
        weights = (flat @ basis.T).reshape(table.shape[:3] + (p.n_basis,))
        residual = np.linalg.norm(shapes - (shapes @ basis.T) @ basis, axis=1)
        kept = float(np.percentile(residual, 99))
        log.info(f"MUAP table: {p.n_basis} waveforms, shape error <= {kept:.3f} for 99 % of the table")
        return {"depth": depth, "theta": theta, "dz": dz_axis, "basis": basis.astype(np.float32),
                "weights": weights.astype(np.float32), "muscle_radius": np.array(cyl.b), "shape_error": np.array(kept)}

    def weights_at(self, depth_m: np.ndarray, theta: np.ndarray, dz_m: np.ndarray) -> np.ndarray:
        """(..., n_basis) waveform weights at these depths below the muscle surface, angles (any sign) and distances
        along the forearm from the end-plate (volts per unit of the fibre source; zero beyond the table). Trilinear
        on the grid (angle and distance axes uniform)."""
        depth_m, theta, dz_m = np.broadcast_arrays(depth_m, theta, dz_m)
        theta = np.abs(np.angle(np.exp(1j * theta)))
        fd = np.interp(depth_m, self.depth, np.arange(len(self.depth)))
        ft = theta / (self.theta[1] - self.theta[0])
        fz = (dz_m - self.dz[0]) / (self.dz[1] - self.dz[0])
        inside = ((fz >= 0) & (fz <= len(self.dz) - 1)).astype(np.float32)
        shape = self.weights.shape[:3]
        index, frac = [], []
        for f, n in ((fd, shape[0]), (ft, shape[1]), (np.clip(fz, 0, shape[2] - 1), shape[2])):
            i0 = np.clip(np.floor(f).astype(int), 0, n - 2)
            index.append(i0)
            frac.append((f - i0).astype(np.float32))
        base = np.ravel_multi_index(index, shape)                    # the 8 corners: base + strides
        strides = np.array([d * shape[1] * shape[2] + t * shape[2] + z for d in (0, 1) for t in (0, 1) for z in (0, 1)])
        (wd, wt, wz) = frac
        w = np.stack([(wd if d else 1 - wd) * (wt if t else 1 - wt) * (wz if z else 1 - wz)
                      for d in (0, 1) for t in (0, 1) for z in (0, 1)], axis=-1) * inside[..., None]
        flat = self.weights.reshape(-1, self.weights.shape[-1])
        return np.einsum("...c,...ck->...k", w, flat[base[..., None] + strides])


def _fibre_current(z: np.ndarray, t: np.ndarray, p: VolumeConductorParams) -> np.ndarray:
    """(t, z) membrane current per unit length (A/m) of a fibre excited at its end-plate (z = 0) at t = 0: the
    Rosenfalck action potential travelling both ways at the conduction velocity, inside the fibre only (its edges
    are where the potential starts and stops: the end-plate and tendon sources)."""
    behind = np.maximum(p.conduction_velocity * t[:, None] - np.abs(z)[None], 0) * 1000 * p.iap_lambda   # mm * lambda
    vm = (96 * behind ** 3 * np.exp(-behind) - 90) / 1000                                               # V
    dz = z[1] - z[0]
    slope = np.gradient(vm, dz, axis=1)
    # The unit's fibres end at scattered tendon junctions: the mean over half-lengths (each side on its own).
    ends = (p.fibre_length_mm / 2 + np.linspace(-1, 1, TENDON_FIBRES) * p.tendon_spread_mm) / 1000
    inside = np.mean([np.where(z >= 0, z <= e, -z <= e) for e in ends], axis=0)
    radius = p.fibre_diameter_um / 2e6
    return p.sigma_intracellular * np.pi * radius ** 2 * np.gradient(slope * inside[None], dz, axis=1)


def _log_i(m, x):
    """log I_m(x) (x > 0); the small-argument limit where the scaled function underflows."""
    m = np.abs(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = special.ive(m, x)
        out = np.log(scaled) + x
    small = m * np.log(x / 2) - special.gammaln(m + 1)
    return np.where(np.isfinite(out) & (scaled > 0), out, small)


def _log_k(m, x):
    """log K_m(x) (x > 0); the small-argument limit where the scaled function overflows."""
    m = np.abs(m)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = special.kve(m, x)
        out = np.log(scaled) - x
    small = special.gammaln(np.maximum(m, 1)) - np.log(2) + m * np.log(2 / x)
    return np.where(np.isfinite(out) & (scaled > 0), out, small)


def _ik(m, x1, x2):
    """I_m(x1) K_m(x2), x1 <= x2."""
    return np.exp(_log_i(m, x1) + _log_k(m, x2))


def _li(m, x):
    """I_m'(x) / I_m(x)."""
    lm = _log_i(m, x)
    return 0.5 * (np.exp(_log_i(m - 1, x) - lm) + np.exp(_log_i(m + 1, x) - lm))


def _lk(m, x):
    """K_m'(x) / K_m(x)."""
    lm = _log_k(m, x)
    return -0.5 * (np.exp(_log_k(m - 1, x) - lm) + np.exp(_log_k(m + 1, x) - lm))


def _ri(m, x1, x2):
    """I_m(x1) / I_m(x2), x1 <= x2."""
    return np.exp(_log_i(m, x1) - _log_i(m, x2))


def _rk(m, x1, x2):
    """K_m(x1) / K_m(x2), x1 >= x2."""
    return np.exp(_log_k(m, x1) - _log_k(m, x2))
