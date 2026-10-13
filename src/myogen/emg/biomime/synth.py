"""Motor-unit-level sEMG from the MUAP bank.

    BioMimeEMG(bank, subject, session, anat)     once per (subject, session): draws each muscle's motor units,
                                                 projects their MUAPs onto this session's electrodes, and
                                                 compresses each channel's templates with an SVD
    .synthesize(u, L, fs_ctrl, rng)              per trial: drive -> spikes (2048 Hz) -> EMG, each spike split
                                                 between the two nearest fibre-length bins

Output is clean sEMG in mV at 2048 Hz; emg.acquisition adds noise and the device, as for the analytic engine.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal, sparse

from .bank import MuapBank
from .electrodes import interp_matrix, skin_filter, subject_channels
from .generator import FS, N_TIME
from .mupool import POOLS, PoolParams

MAX_UNITS_PER_MUSCLE = 96      # drawn from the bank's 128 per muscle: subject-specific subsets
SVD_ENERGY = 0.999
SVD_MAX_RANK = 32


@dataclass
class UnitTable:
    """Ground truth for every simulated motor unit (decomposition validation, SNN training)."""
    muscle: np.ndarray       # (N,) index into BioMimeEMG.muscles
    rank: np.ndarray         # (N,) position in the full pool (0 smallest .. 1 largest)
    threshold: np.ndarray    # (N,) recruitment drive
    size: np.ndarray         # (N,) fibres
    cond: np.ndarray         # (N, 6) BioMime conditions at L = 1
    weight: np.ndarray       # (N,) amplitude factor applied (pool subsampling, depth, axial overlap, subject)


class BioMimeEMG:
    """Motor-unit EMG for one subject / session (see the module docstring).

    gain: global amplitude calibration (1 = physical BioMime amplitudes). Absolute sEMG amplitude is not pinned
    down by the model; fit it against a target device with `myogen realism`.
    """

    def __init__(self, bank: MuapBank, subject, session, anat, pool="fuglevand", pool_params=None,
                 units_per_muscle=MAX_UNITS_PER_MUSCLE, ied=None, keep_templates=False, gain=1.0):
        self.bank, self.anat = bank, anat
        self.muscles = [m for m in bank.muscles if m in anat.muscle_names]
        self.model_index = np.array([anat.muscle_names.index(m) for m in self.muscles])
        P = self._electrodes(subject, session, ied)
        H = self._units(subject, P, POOLS[pool], pool_params or PoolParams(), units_per_muscle, gain)
        self.n_units, self.n_bins, self.n_channels = H.shape[:3]
        self.H = H if keep_templates else None       # (N, nL, C, 96) mV, for exact reference synthesis
        self._factorise(H)

    def _electrodes(self, subject, session, ied):
        """This session's channels; returns their (C, 320) weights on the grid, through the subject's skin."""
        territory_angle = {m: tuple(self.bank.meta[m]["angle"]) for m in self.muscles}
        self.channels = subject_channels(subject.layout, session, self.anat.forearm_length, territory_angle,
                                         ied=ied or subject.device_params.get("ied", 0.02),
                                         rng=np.random.default_rng(subject.seed + 3))
        W, self.poles_outside = interp_matrix(self.channels, self.bank.grid)
        return W @ skin_filter(self.bank.grid, subject.fat)

    def _units(self, subject, P, Pool, params, units_per_muscle, gain):
        """Draw each muscle's motor units (a subject-stable subset of the bank), build their pools and project
        their MUAPs onto the channels. Returns the templates (N, nL, C, 96) in mV."""
        rng = np.random.default_rng(subject.seed + 17)
        self.pools, templates, unit_rows = [], [], []
        first_unit = 0
        for j, muscle in enumerate(self.muscles):
            meta, table = self.bank.meta[muscle], self.bank.unit_table(muscle)
            n_bank = len(table["size"])
            n_sel = min(units_per_muscle, n_bank)
            sel = np.sort(rng.choice(n_bank, n_sel, replace=False))
            w = _muscle_weight(meta, subject, self.model_index[j], n_sel) * gain
            if w <= 1e-4:
                continue
            rank = table["pool_index"][sel] / max(meta["n_mu"] - 1, 1)
            pool = Pool(rank, params)
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

    def synthesize(self, u, L, fs_ctrl, rng, return_spikes=False, exact=False):
        """u, L: (T, nu_model) excitation and normalised fibre length. Returns (N, C) mV at 2048 Hz."""
        T = u.shape[0]
        N = int(round(T / fs_ctrl * FS))
        t_ctrl = np.arange(T) / fs_ctrl
        t = np.arange(N) / FS
        L_bins = self.bank.L_bins
        rows, cols, vals = [], [], []
        spike_log = []
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
                if return_spikes:
                    spike_log.append((np.full(len(s), unit, np.int32), s.astype(np.int32)))
        if rows:
            S = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(N, self.n_units * self.n_bins), dtype=np.float32)
        else:
            S = sparse.csr_matrix((N, self.n_units * self.n_bins), dtype=np.float32)
        K, C = self.A.shape[1:]
        if exact:
            if self.H is None:
                raise ValueError("exact synthesis needs keep_templates=True")
            X = S @ self.H.reshape(-1, C * N_TIME)                      # (N, C*96) sparse spikes x templates
            emg = np.zeros((N + N_TIME, C))
            Xr = X.reshape(N, C, N_TIME)
            for tau in range(N_TIME):                                  # shift-and-add (reference only)
                emg[tau:tau + N] += Xr[:, :, tau]
            emg = emg[:N]
        else:
            X = (S @ self.A.reshape(-1, K * C)).reshape(N, K, C)
            emg = signal.fftconvolve(X, self.Phi, axes=0)[:N].sum(1)      # MUAP onset at the spike
        if return_spikes:
            units = np.concatenate([a for a, _ in spike_log]) if spike_log else np.zeros(0, np.int32)
            times = np.concatenate([b for _, b in spike_log]) if spike_log else np.zeros(0, np.int32)
            return emg.astype(np.float32), (units, times)
        return emg.astype(np.float32)

    def spike_table(self, spikes):
        """Spikes (unit, sample @ 2048 Hz) and the unit table, keyed by their HDF5 dataset names."""
        unit, sample = spikes
        table = {"unit": unit, "sample": sample}
        table.update({f"units/{k}": getattr(self.units, k)
                      for k in ("muscle", "rank", "threshold", "size", "cond", "weight")})
        return table

    def summary(self):
        return {"units": int(self.n_units), "muscles": len(self.pools), "channels": int(self.n_channels),
                "svd_rank": self.rank_used.tolist(), "svd_energy": np.round(self.energy, 4).tolist(),
                "poles_outside_grid": int(self.poles_outside)}


def _muscle_weight(meta, subject, model_index, n_sel):
    """Amplitude of one muscle's units: keeps the muscle's EMG power with fewer simulated units, the bank gain
    (depth beyond BioMime's range, axial overlap with the grid), the subject's muscle size / atrophy, what is left
    of it in a residual limb, and the subject's overall amplitude (gain_mv / 1.5: median 1)."""
    z_lo, z_hi = meta["z_range"]
    L_res = subject.residual_length
    alive = np.clip((L_res - z_lo) / max(z_hi - z_lo, 1e-3), 0, 1) if L_res < 1 else 1.0
    return np.sqrt(meta["n_mu"] / n_sel) * meta["gain"] * subject.muscle_amp[model_index] * alive \
        * subject.gain_mv / 1.5
