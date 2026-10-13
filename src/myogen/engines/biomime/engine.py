"""BioMimeEngine: motor-unit-level sEMG from the MUAP bank.

    prepare   draws each muscle's motor units from the bank (a subject-specific subset), projects their MUAPs onto
              the electrodes through the subject's skin, and compresses each channel's templates with an SVD
    generate  drive -> motor-unit spikes (2048 Hz); each spike adds its MUAP, interpolated between the two nearest
              fibre-length bins (MUAPs change as muscles shorten)

Electrodes are poles on BioMime's 10 x 32 monopolar grid (bilinear interpolation). The grid covers the whole
circumference but only part of the forearm's length (GridGeometry); electrodes beyond it are rejected.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal, sparse

from .. import CleanEMG, EMGEngine
from .anatomy import FAT_REF, GridGeometry, fit_angle_transform
from .bank import DEFAULT_BANK, MuapBank
from .network import FS, N_COL, N_ROW, N_TIME
from .pools import POOLS, PoolParams

MAX_UNITS_PER_MUSCLE = 96      # drawn from the bank's 128 per muscle: subject-specific subsets
SVD_ENERGY = 0.999
SVD_MAX_RANK = 32
FAT_ATTEN_DECAY = 0.006        # m, extra attenuation for fat beyond BioMime's reference (its own depth decay)


@dataclass
class UnitTable:
    """Ground truth for every simulated motor unit (decomposition validation, SNN training)."""
    muscle: np.ndarray       # (N,) index into BioMimeEngine.muscles
    rank: np.ndarray         # (N,) position in the full pool (0 smallest .. 1 largest)
    threshold: np.ndarray    # (N,) recruitment drive
    size: np.ndarray         # (N,) fibres
    cond: np.ndarray         # (N, 6) BioMime conditions at L = 1
    weight: np.ndarray       # (N,) amplitude factor applied (pool subsampling, depth, axial overlap, subject)


class BioMimeEngine(EMGEngine):
    name = "biomime"

    def __init__(self, muap_bank=None, pool="fuglevand", gain=1.0, save_spikes=False,
                 units_per_muscle=MAX_UNITS_PER_MUSCLE, keep_templates=False, fs=2000.0):
        """muap_bank: path of the bank (default: DEFAULT_BANK, see `myogen biomime-setup`); pool: fuglevand | lif;
        gain: global amplitude (1 = physical BioMime amplitudes; absolute sEMG amplitude depends on contact and
        effort the model does not pin down); keep_templates: allow exact (slow) reference synthesis."""
        if pool not in POOLS:
            raise ValueError(f"pool must be {' | '.join(POOLS)}, got {pool!r}")
        self.bank_path, self.pool, self.gain, self.save_spikes = muap_bank or DEFAULT_BANK, pool, gain, save_spikes
        self.units_per_muscle, self.keep_templates, self.fs = units_per_muscle, keep_templates, fs

    def prepare(self, model, subject, electrodes):
        self.bank = bank = MuapBank(self.bank_path)
        self.muscles = [m for m in bank.muscles if m in model.muscle_names]
        self.model_index = np.array([model.muscle_names.index(m) for m in self.muscles])
        P = self._projection(model, subject, electrodes)                  # (C, 320) grid -> channels
        H = self._units(model, subject, P)                                # (N, nL, C, 96) mV
        self.n_units, self.n_bins, self.n_channels = H.shape[:3]
        self.H = H if self.keep_templates else None
        self._factorise(H)

    def generate(self, activity, rng, exact=False):
        emg, spikes = self.synthesize(activity.drive, activity.fibre_length, activity.fs, rng, exact=exact)
        g = np.gcd(int(self.fs), FS)
        emg = signal.resample_poly(emg, int(self.fs) // g, FS // g, axis=0) if self.fs != FS else emg
        return CleanEMG(emg.astype(np.float32), self.fs, self.spike_table(spikes) if self.save_spikes else None)

    # -- set-up ------------------------------------------------------------------------------------------------

    def _projection(self, model, subject, electrodes):
        """(C, 320): each electrode's two poles interpolated on the grid, then the subject's skin filter."""
        grid = self.bank.grid
        offset, _ = fit_angle_transform(model)                 # our forearm-frame angle -> BioMime angle
        circumference = grid.circumference_ref
        L = model.forearm_length
        lo, hi = grid.z0_frac * L * 100, grid.z1_frac * L * 100
        W = np.zeros((len(electrodes), N_ROW * N_COL))
        for c, e in enumerate(electrodes.electrodes):
            # Poles on BioMime's reference forearm: arc lengths scale with its circumference.
            ends = e.poles(lambda z: circumference / (2 * np.pi))
            for (angle, z), weight in zip(ends, (1.0, -1.0)):
                if not lo - 1e-9 <= z * 100 <= hi + 1e-9:
                    raise ValueError(f"electrode {e.name}: a pole at {z * 100:.1f} cm from the elbow is outside the "
                                     f"BioMime grid ({lo:.2f}-{hi:.2f} cm); move it or use engine='analytic'")
                W[c] += weight * _bilinear((angle + offset) / (2 * np.pi), grid.row_of(z / L))
        return W @ _skin_filter(grid, subject.fat_mm / 1000)

    def _units(self, model, subject, P):
        """Draw each muscle's motor units, build their pools and project their MUAPs onto the channels."""
        rng = np.random.default_rng(subject.seed + 17)
        amp = subject.muscle_amplitudes(model.extrinsic)
        self.pools, templates, unit_rows = [], [], []
        first_unit = 0
        for j, muscle in enumerate(self.muscles):
            meta, table = self.bank.meta[muscle], self.bank.unit_table(muscle)
            n_bank = len(table["size"])
            n_sel = min(self.units_per_muscle, n_bank)
            sel = np.sort(rng.choice(n_bank, n_sel, replace=False))
            w = _muscle_weight(meta, subject, amp[self.model_index[j]], n_sel) * self.gain
            if w <= 1e-4:
                continue
            rank = table["pool_index"][sel] / max(meta["n_mu"] - 1, 1)
            pool = POOLS[self.pool](rank, PoolParams())
            self.pools.append((j, pool, first_unit))
            muap = self.bank.muaps(muscle, sel)                             # (n, nL, 10, 32, 96) uV
            n, nL = muap.shape[:2]
            unit_w = w * table["amp_extra"][sel] * 1e-3                     # -> mV
            proj = np.tensordot(muap.reshape(n, nL, -1, N_TIME), P, axes=([2], [1]))   # (n, nL, 96, C)
            templates.append(np.ascontiguousarray(proj.transpose(0, 1, 3, 2)) * unit_w[:, None, None, None])
            unit_rows.append((np.full(n, j), rank, pool.rt, table["size"][sel], table["cond"][sel], unit_w * 1e3))
            first_unit += n
        self.units = UnitTable(*(np.concatenate(column) for column in zip(*unit_rows)))
        return np.concatenate(templates)

    def _factorise(self, H):
        """Low rank per channel, H[:, :, c, :] ~ A_c @ Phi_c, keeping SVD_ENERGY of the energy."""
        K, C = SVD_MAX_RANK, self.n_channels
        self.A = np.zeros((self.n_units * self.n_bins, K, C), np.float32)
        self.Phi = np.zeros((N_TIME, K, C), np.float32)
        self.rank_used, self.energy = np.zeros(C, int), np.zeros(C)
        for c in range(C):
            # Principal waveforms from the 96 x 96 Gram matrix (same basis as the SVD, far cheaper).
            M = H[:, :, c, :].reshape(-1, N_TIME).astype(np.float64)
            ev, V = np.linalg.eigh(M.T @ M)
            ev, V = ev[::-1].clip(min=0), V[:, ::-1]
            e = np.cumsum(ev) / max(ev.sum(), 1e-30)
            k = int(min(np.searchsorted(e, SVD_ENERGY) + 1, K))
            self.A[:, :k, c] = M @ V[:, :k]
            self.Phi[:, :k, c] = V[:, :k]
            self.rank_used[c], self.energy[c] = k, e[k - 1]

    # -- synthesis ---------------------------------------------------------------------------------------------

    def synthesize(self, u, L, fs_ctrl, rng, exact=False):
        """u, L: (T, nu) drive and normalised fibre length at fs_ctrl. Returns ((N, C) mV at 2048 Hz, spikes)."""
        T = u.shape[0]
        N = int(round(T / fs_ctrl * FS))
        t_ctrl, t = np.arange(T) / fs_ctrl, np.arange(N) / FS
        L_bins = self.bank.L_bins
        rows, cols, vals, spike_log = [], [], [], []
        for j, pool, first_unit in self.pools:
            mi = self.model_index[j]
            drive = np.interp(t, t_ctrl, np.clip(u[:, mi], 0, 1))
            beta = np.interp(np.interp(t, t_ctrl, L[:, mi]), L_bins, np.arange(len(L_bins)))
            lo = np.minimum(np.floor(beta).astype(int), len(L_bins) - 2)
            fr = beta - lo
            for i, s in enumerate(pool.spikes(drive, FS, rng)):
                if len(s) == 0:
                    continue
                unit = first_unit + i
                base = unit * self.n_bins + lo[s]
                rows += [s, s]
                cols += [base, base + 1]
                vals += [1 - fr[s], fr[s]]
                spike_log.append((np.full(len(s), unit, np.int32), s.astype(np.int32)))
        shape = (N, self.n_units * self.n_bins)
        S = (sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape,
                               dtype=np.float32) if rows else sparse.csr_matrix(shape, dtype=np.float32))
        K, C = self.A.shape[1:]
        if exact:
            if self.H is None:
                raise ValueError("exact synthesis needs keep_templates=True")
            X = (S @ self.H.reshape(-1, C * N_TIME)).reshape(N, C, N_TIME)
            emg = np.zeros((N + N_TIME, C))
            for tau in range(N_TIME):                                  # shift-and-add (reference only)
                emg[tau:tau + N] += X[:, :, tau]
            emg = emg[:N]
        else:
            X = (S @ self.A.reshape(-1, K * C)).reshape(N, K, C)
            emg = signal.fftconvolve(X, self.Phi, axes=0)[:N].sum(1)      # MUAP onset at the spike
        units = np.concatenate([a for a, _ in spike_log]) if spike_log else np.zeros(0, np.int32)
        times = np.concatenate([b for _, b in spike_log]) if spike_log else np.zeros(0, np.int32)
        return emg.astype(np.float32), (units, times)

    def spike_table(self, spikes):
        """Spikes (unit, sample @ 2048 Hz) and the unit table, keyed by their HDF5 dataset names."""
        unit, sample = spikes
        table = {"unit": unit, "sample": sample}
        table.update({f"units/{k}": getattr(self.units, k)
                      for k in ("muscle", "rank", "threshold", "size", "cond", "weight")})
        return table


def _bilinear(angle_frac, row):
    """(320,) weights of one point on the flattened (row, col) grid; the circumference wraps around."""
    w = np.zeros(N_ROW * N_COL)
    col = (angle_frac % 1.0) * N_COL
    r0, c0 = min(int(np.floor(row)), N_ROW - 2), int(np.floor(col))
    fr, fc = row - r0, col - c0
    for dr, wr in ((0, 1 - fr), (1, fr)):
        for dc, wc in ((0, 1 - fc), (1, fc)):
            w[(r0 + dr) * N_COL + (c0 + dc) % N_COL] += wr * wc
    return w


def _skin_filter(grid: GridGeometry, fat):
    """(320, 320) spatial low-pass x attenuation for fat thicker than BioMime's reference forearm (no fat
    condition in the network)."""
    extra = max(0.0, fat - FAT_REF)
    if extra <= 0:
        return np.eye(N_ROW * N_COL)
    rows = np.arange(N_ROW) * grid.row_pitch
    cols = np.arange(N_COL)
    dcol = np.minimum(np.abs(cols[:, None] - cols[None, :]), N_COL - np.abs(cols[:, None] - cols[None, :]))
    Kr = np.exp(-0.5 * ((rows[:, None] - rows[None, :]) / extra) ** 2)
    Kc = np.exp(-0.5 * (dcol * grid.col_pitch / extra) ** 2)
    Kr /= Kr.sum(1, keepdims=True)
    Kc /= Kc.sum(1, keepdims=True)
    return np.kron(Kr, Kc) * np.exp(-extra / FAT_ATTEN_DECAY)


def _muscle_weight(meta, subject, muscle_amp, n_sel):
    """Amplitude of one muscle's units: keeps the muscle's EMG power with fewer simulated units, the bank gain
    (depth beyond BioMime's range, axial overlap with the grid), the subject's muscle size, what is left of it in
    a residual limb, and the subject's overall amplitude (gain_mv / 1.5: median 1)."""
    z_lo, z_hi = meta["z_range"]
    L_res = subject.residual_length
    alive = np.clip((L_res - z_lo) / max(z_hi - z_lo, 1e-3), 0, 1) if L_res < 1 else 1.0
    return np.sqrt(meta["n_mu"] / n_sel) * meta["gain"] * muscle_amp * alive * subject.gain_mv / 1.5
