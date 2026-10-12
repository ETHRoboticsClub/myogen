"""Motor-unit-level sEMG from the MUAP bank: spikes x dynamic MUAPs, per subject / session / trial.

  BioMimeEMG(bank, subject, session, anat)        once per (subject, session)
      * draws each muscle's motor units from the bank (subject-specific subset, size-ordered pools)
      * projects their MUAPs (all fibre-length bins) onto this session's electrodes through the skin filter
      * compresses each channel's templates with an SVD (rank chosen per channel for 99.5 % energy)
  .synthesize(u, L, fs_ctrl, rng)                  per trial
      * drive u (excitation, 0-1) and normalised fibre length L per muscle at the control rate
      * motor-unit pools -> spike times (2048 Hz)
      * each spike is split between the two nearest fibre-length bins (dynamic MUAP interpolation)
      * EMG_c = sum_k phi_ck * (S @ A_c)_k : one sparse product and one batched FFT convolution per trial,
        exact up to the SVD truncation (vs. one convolution per motor unit and channel)

Output is clean monopolar-derived sEMG in mV at 2048 Hz; src.emg.acquire adds noise, mains, artifacts and the
device front-end, exactly as for the analytic model.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal, sparse

from .bank import MuapBank
from .electrodes import interp_matrix, skin_filter, subject_channels
from .model import FS, N_TIME
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
    def __init__(self, bank: MuapBank, subject, session, anat, pool="fuglevand", pool_params=None,
                 units_per_muscle=MAX_UNITS_PER_MUSCLE, ied=None, keep_templates=False):
        self.bank, self.anat = bank, anat
        rng = np.random.default_rng(subject.seed + 17)              # subject-stable unit selection
        self.muscles = [m for m in bank.muscles if m in anat.muscle_names]
        self.model_index = np.array([anat.muscle_names.index(m) for m in self.muscles])
        params = pool_params or PoolParams()
        L_res = subject.residual_length

        # Electrodes for this session, through the subject's skin.
        territory_angle = {m: tuple(bank.meta[m]["angle"]) for m in self.muscles}
        ied = ied or subject.device_params.get("ied", 0.02)
        self.channels = subject_channels(subject.layout, session, anat.forearm_length, territory_angle,
                                         ied=ied, rng=np.random.default_rng(subject.seed + 3))
        W, self.poles_outside = interp_matrix(self.channels, bank.grid)
        P = W @ skin_filter(bank.grid, subject.fat)                         # (C, 320)
        C = P.shape[0]

        self.pools, templates, rows = [], [], []
        mus, ranks, ths, sizes, conds, weights = [], [], [], [], [], []
        for j, m in enumerate(self.muscles):
            meta = bank.meta[m]
            table = bank.unit_table(m)
            n_bank = len(table["size"])
            n_sel = min(units_per_muscle, n_bank)
            sel = np.sort(rng.choice(n_bank, n_sel, replace=False))
            # Amplitude: keep the muscle's total EMG power with fewer simulated units; bank gain covers depth
            # beyond BioMime's range and axial overlap with the grid; subject factor covers size/atrophy.
            z_lo, z_hi = meta["z_range"]
            alive = np.clip((L_res - z_lo) / max(z_hi - z_lo, 1e-3), 0, 1) if L_res < 1 else 1.0
            # subject.gain_mv / 1.5: between-subject amplitude spread (skin impedance, anatomy), median 1.
            w = np.sqrt(meta["n_mu"] / n_sel) * meta["gain"] * subject.muscle_amp[self.model_index[j]] * alive \
                * subject.gain_mv / 1.5
            if w <= 1e-4:
                continue
            rank = table["pool_index"][sel] / max(meta["n_mu"] - 1, 1)
            mu_pool = POOLS[pool](rank, params)
            self.pools.append((j, mu_pool, len(templates)))
            muap = bank.muaps(m, sel)                                       # (n, nL, 10, 32, 96) uV
            n, nL = muap.shape[:2]
            unit_w = w * table["amp_extra"][np.sort(sel)] * 1e-3            # -> mV
            proj = np.tensordot(muap.reshape(n, nL, -1, N_TIME), P, axes=([2], [1]))   # (n, nL, 96, C)
            proj = np.ascontiguousarray(proj.transpose(0, 1, 3, 2)) * unit_w[:, None, None, None]
            templates.append(proj)
            mus.append(np.full(n, j)), ranks.append(rank), ths.append(mu_pool.rt), sizes.append(table["size"][sel])
            conds.append(table["cond"][sel]), weights.append(unit_w * 1e3)
        H = np.concatenate(templates)                                       # (N, nL, C, 96)
        self.n_units, self.n_bins = H.shape[:2]
        self.units = UnitTable(np.concatenate(mus), np.concatenate(ranks), np.concatenate(ths),
                               np.concatenate(sizes), np.concatenate(conds), np.concatenate(weights))
        self.unit_offset = np.cumsum([0] + [t.shape[0] for t in templates])
        self.H = H if keep_templates else None       # (N, nL, C, 96) mV, for exact reference synthesis

        # Low-rank factorisation per channel: H[:, :, c, :] ~ A_c @ Phi_c
        K = SVD_MAX_RANK
        self.A = np.zeros((self.n_units * self.n_bins, K, C), np.float32)
        self.Phi = np.zeros((N_TIME, K, C), np.float32)
        self.rank_used, self.energy = np.zeros(C, int), np.zeros(C)
        for c in range(C):
            # Principal waveforms from the 96x96 Gram matrix (same basis as the SVD of M, far cheaper).
            M = H[:, :, c, :].reshape(-1, N_TIME).astype(np.float64)
            ev, V = np.linalg.eigh(M.T @ M)
            ev, V = ev[::-1].clip(min=0), V[:, ::-1]
            e = np.cumsum(ev) / max(ev.sum(), 1e-30)
            k = int(min(np.searchsorted(e, SVD_ENERGY) + 1, K))
            self.A[:, :k, c] = M @ V[:, :k]
            self.Phi[:, :k, c] = V[:, :k]
            self.rank_used[c], self.energy[c] = k, e[k - 1]
        self.n_channels = C

    def synthesize(self, u, L, fs_ctrl, rng, return_spikes=False, exact=False):
        """u, L: (T, nu_model) excitation and normalised fibre length. Returns (N, C) mV at 2048 Hz."""
        T = u.shape[0]
        N = int(round(T / fs_ctrl * FS))
        t_ctrl = np.arange(T) / fs_ctrl
        t = np.arange(N) / FS
        L_bins = self.bank.L_bins
        rows, cols, vals = [], [], []
        spike_log = []
        for j, pool, k0 in self.pools:
            mi = self.model_index[j]
            drive = np.interp(t, t_ctrl, np.clip(u[:, mi], 0, 1))
            beta = np.interp(np.interp(t, t_ctrl, L[:, mi]), L_bins, np.arange(len(L_bins)))
            lo = np.minimum(np.floor(beta).astype(int), len(L_bins) - 2)
            fr = beta - lo
            for i, s in enumerate(pool.spikes(drive, FS, rng)):
                if len(s) == 0:
                    continue
                unit = self.unit_offset[k0] + i
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

    def summary(self):
        return {"units": int(self.n_units), "muscles": len(self.pools), "channels": int(self.n_channels),
                "svd_rank": self.rank_used.tolist(), "svd_energy": np.round(self.energy, 4).tolist(),
                "poles_outside_grid": int(self.poles_outside)}
