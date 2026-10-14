from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("myogen")
except PackageNotFoundError:
    __version__ = "unknown"

from .inputs.electrodes import Electrode, Electrodes, Frontend
from .inputs.motion import HandMotion, JointAdapter, example_motion
from .inputs.subject import Subject
from .params import SimParams
from .recording import Recording
from .simulator import Simulator
from .tools.calibration import Calibration, calibrate, compare


def replay(recording: Recording | str | Path, port: int = 8080, emg_window: float = 4.0) -> None:
    """Open a Recording (or a recording file) in the web viewer (needs myogen[viewer])."""
    from .tools.viewer import replay as _replay
    return _replay(recording, port=port, emg_window=emg_window)


__all__ = ["Calibration", "calibrate", "compare", "Electrode", "Electrodes", "Frontend", "HandMotion", "JointAdapter",
           "Recording", "SimParams", "Simulator", "Subject", "example_motion", "replay"]
