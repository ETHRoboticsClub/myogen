"""EMG engines: muscle activity + electrode positions -> clean bipolar sEMG. Both share one interface:

    engine.prepare(model, subject, electrodes)      once: geometry, sources / motor units
    engine.generate(activity, rng) -> CleanEMG      per run

    AnalyticEngine   interference-pattern model: band-limited noise sources along each muscle belly, attenuated
                     with distance to each pole; fast, no extra inputs
    BioMimeEngine    motor-unit model: spikes x MUAPs from the BioMime network (Ma et al. 2023); needs PyTorch
                     and a MUAP bank built from the published weights (`myogen biomime-setup`)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class CleanEMG:
    emg: np.ndarray                 # (N, C) bipolar sEMG in mV, before noise and the device
    fs: float
    spikes: dict | None = None      # BioMime: motor-unit spike trains and the unit table


class EMGEngine(ABC):
    name = ""

    @abstractmethod
    def prepare(self, model, subject, electrodes):
        """Set up for this subject and these electrodes (called once by the Simulator)."""

    @abstractmethod
    def generate(self, activity, rng) -> CleanEMG:
        """Clean sEMG for a MuscleActivity."""


def make_engine(engine, **options):
    """An engine from its name ("analytic" | "biomime") or an EMGEngine instance."""
    if isinstance(engine, EMGEngine):
        return engine
    if engine == "analytic":
        from .analytic import AnalyticEngine
        return AnalyticEngine(**options)
    if engine == "biomime":
        from .biomime import BioMimeEngine
        return BioMimeEngine(**options)
    raise ValueError(f"engine must be analytic | biomime, got {engine!r}")
