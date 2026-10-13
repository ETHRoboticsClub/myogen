"""myogen: synthetic forearm sEMG from hand and wrist joint angles, on a musculoskeletal forearm model.

    import myogen as mg
    motion = mg.HandMotion.load("motion.h5")                    # or mg.JointAdapter(mapping)(angles, fs)
    electrodes = mg.Electrodes.ring(n=8, position_cm=7)         # or .myo(), .otto_bock(), your own Electrode list
    recording = mg.Simulator(electrodes, engine="biomime").run(motion)   # engine: "analytic" | "biomime"
    recording.save("emg.h5")
    mg.replay(recording)                                        # web viewer: moving hand + EMG
    mg.calibrate(simulator, motions, real_emgs)                 # fit a device profile to real recordings
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("myogen")
except PackageNotFoundError:
    __version__ = "unknown"

from .calibration import Calibration, calibrate, compare
from .electrodes import Electrode, Electrodes, Frontend
from .motion import HandMotion, JointAdapter, example_motion
from .recording import Recording
from .simulator import Simulator
from .subject import Subject


def replay(recording, port=8080, emg_window=4.0):
    """Open a Recording (or a recording file) in the web viewer (needs myogen[viewer])."""
    from .viewer import replay as _replay
    return _replay(recording, port=port, emg_window=emg_window)


__all__ = ["Calibration", "calibrate", "compare", "Electrode", "Electrodes", "Frontend", "HandMotion", "JointAdapter", "Recording", "Simulator", "Subject",
           "example_motion", "replay"]
