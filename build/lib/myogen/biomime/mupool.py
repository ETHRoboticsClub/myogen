"""Motor-unit pools: neural drive -> spike trains, one interface for rate models and spiking neurons.

All pools follow the size principle: units are ordered by size (fibres), recruitment thresholds increase
exponentially with rank, and spikes are returned as sample indices per unit so they can be convolved with MUAPs,
stored for decomposition ground truth, or used to train / drive spiking neural networks.

  FuglevandPool  rate model (Fuglevand et al. 1993; defaults as in NeuroMotion / iEMG): firing rate grows
                 linearly above threshold up to an onion-skin peak rate; inter-spike intervals are a renewal
                 process with Gaussian jitter. Fully vectorised (cumulative phase + searchsorted).
  LIFPool        leaky integrate-and-fire motoneurons (input resistance ~ 1 / threshold, membrane time
                 constant, refractory period, noisy input); the place to plug in richer SNN dynamics.

Add a new model by subclassing MotorUnitPool and implementing `spikes(drive, fs, rng)`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoolParams:
    rr: float = 50.0        # range of recruitment thresholds (last / first)
    rm: float = 0.75        # drive at which the last unit is recruited
    pfr1: float = 40.0      # peak firing rate of the first unit (Hz)
    pfrd: float = 10.0      # first - last peak firing rate (onion skin)
    mfr: float = 8.0        # firing rate at recruitment (Hz)
    gain: float = 30.0      # Hz per unit of drive above threshold
    isi_cv: float = 0.15    # inter-spike-interval coefficient of variation
    extra: dict = field(default_factory=dict)


class MotorUnitPool(ABC):
    def __init__(self, rank, params: PoolParams = None):
        """rank: (n,) position of each simulated unit in the full pool, 0 = smallest/first recruited, 1 = largest."""
        self.p = params or PoolParams()
        self.rank = np.asarray(rank, np.float64)
        self.n = len(self.rank)
        self.rt = self.p.rm / self.p.rr * self.p.rr ** self.rank           # exponential thresholds
        self.pfr = self.p.pfr1 - self.p.pfrd * self.rank

    @abstractmethod
    def spikes(self, drive, fs, rng):
        """drive: (T,) in [0, 1] at fs. Returns a list of n int arrays of spike sample indices."""

    def properties(self):
        return {"rank": self.rank, "threshold": self.rt, "peak_rate": self.pfr}


class FuglevandPool(MotorUnitPool):
    def rates(self, drive):
        """(T, n) instantaneous firing rates."""
        e = np.asarray(drive, np.float64)[:, None]
        fr = np.minimum(self.p.mfr + self.p.gain * (e - self.rt[None]), self.pfr[None])
        return np.where(e >= self.rt[None], fr, 0.0)

    def spikes(self, drive, fs, rng):
        fr = self.rates(drive)
        phase = np.cumsum(fr, axis=0) / fs                         # expected number of spikes so far
        out = []
        for i in range(self.n):
            total = phase[-1, i]
            if total < 0.5:
                out.append(np.zeros(0, np.int64))
                continue
            k = int(total * 1.5) + 4
            isi = np.maximum(1.0 + self.p.isi_cv * rng.standard_normal(k), 0.2)
            thresholds = rng.uniform(0, 1) + np.cumsum(isi) - isi[0]   # random phase at first recruitment
            idx = np.searchsorted(phase[:, i], thresholds[thresholds < total])
            out.append(idx.astype(np.int64))
        return out


class LIFPool(MotorUnitPool):
    """Leaky integrate-and-fire: tau dv/dt = -v + I(t) + noise; spike and reset at v >= 1; refractory period.

    The input is shaped so recruitment happens at the same thresholds as the rate model and the steady-state
    voltage spans ~1.0-1.0+v_span over each unit's operating range, where the LIF frequency-current curve
    covers ~8-40 Hz with tau = 20 ms. Membrane noise makes discharges irregular (ISI CV ~0.1-0.2).
    """

    def __init__(self, rank, params: PoolParams = None, tau=0.02, t_ref=0.003, v_span=0.3, noise=0.08):
        super().__init__(rank, params)
        self.tau, self.t_ref, self.v_span, self.noise = tau, t_ref, v_span, noise

    def input_current(self, drive):
        e = np.asarray(drive, np.float64)[:, None]
        above = 1.0 + self.v_span * (e - self.rt) / np.maximum(1.0 - self.rt, 0.25)
        return np.where(e >= self.rt, above, e / self.rt)

    def spikes(self, drive, fs, rng):
        dt = 1.0 / fs
        a = np.exp(-dt / self.tau)
        I = self.input_current(drive)
        v = np.zeros(self.n)
        ref = np.zeros(self.n)
        idx = [[] for _ in range(self.n)]
        sig = self.noise * np.sqrt(1 - a * a)
        for t in range(len(I)):
            v = a * v + (1 - a) * I[t] + sig * rng.standard_normal(self.n)
            v[ref > 0] = 0.0
            ref -= dt
            fired = v >= 1.0
            if fired.any():
                for i in np.flatnonzero(fired):
                    idx[i].append(t)
                v[fired] = 0.0
                ref[fired] = self.t_ref
        return [np.asarray(s, np.int64) for s in idx]


POOLS = {"fuglevand": FuglevandPool, "lif": LIFPool}
