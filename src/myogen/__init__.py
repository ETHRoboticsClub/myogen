"""myogen: synthetic forearm sEMG from a hand/wrist movement recording, on a musculoskeletal forearm model.

    import myogen as mg
    motion = mg.Motion.load("motion.h5")             # or mg.Motion(latent=z, fs=100), or mg.example_motion()
    cfg = mg.SimConfig.load("sim.yml")               # or mg.SimConfig(engine="biomime", device="myo")
    result = mg.simulate(motion, cfg, out="emg.h5")

Package layout, in pipeline order:
    motion/      posture latent, movement recordings, synthetic and NinaPro motion
    body/        forearm model, external loads, muscle controller
    emg/         analytic and BioMime EMG engines, recording chain (noise, devices)
    subject.py   randomised subject / session parameters
    simulation/  one trial, SimConfig, simulate()
    dataset/     `myogen generate`, realism checks, downloads
    viewer/      web viewer, live view, PNG plots
    cli.py, commands.py, config.py   the `myogen` command line (mg.run) and its YAML config
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("myogen")
except PackageNotFoundError:
    __version__ = "unknown"

from .commands import COMMANDS, run
from .config import load_config
from .motion import LATENT_NAMES, Motion, example_motion
from .simulation import SimConfig, SimResult, simulate

__all__ = ["LATENT_NAMES", "Motion", "SimConfig", "SimResult", "example_motion", "simulate",
           "COMMANDS", "load_config", "run"]
