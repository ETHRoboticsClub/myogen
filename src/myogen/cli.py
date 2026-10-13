"""myogen: synthetic forearm sEMG data generation.

  simulate MOTION OUT    one movement recording -> one sEMG recording (myogen simulate -h)

Dataset tools (parameters in a YAML config):

  command                what                                                config
  project
    init [DIR]           write a myogen.yml extending the defaults (in DIR or here)
    config               print the resolved configuration
  setup
    download [WHAT ...]  latent-bank | muap-bank | biomime-weights | all     paths, download
    build-model          rebuild the packaged musculoskeletal model from MyoArm
    ninapro-bank         latent bank from raw NinaPro glove data             paths.ninapro, paths.latent_bank
    biomime-bank ACTION  BioMime MUAP bank: build | info | regrid            biomime_bank, paths.muap_bank
  data
    generate             generate a dataset (live_view: watch it)            generate, electrodes.<device>
    realism              compare generated EMG with NinaPro                  realism
  inspect
    view                 3D replay with setup + live EMG | demo | live UDP   view
    inspect              PNG of one trial                                    inspect
    inspect-biomime      spike raster + EMG + drive of a BioMime trial       inspect
    inspect-muaps        bank MUAPs on BioMime's 10 x 32 grid                inspect

Config: --config FILE > $MYOGEN_CONFIG > ./myogen.yml > the packaged defaults (myogen config -c default).
Data (downloads, NinaPro): paths.data, default $MYOGEN_DATA or ~/.local/share/myogen.
Overrides: any number of KEY=VALUE after the command (VALUE is YAML), e.g.
  myogen generate generate.subjects=4 generate.device=myo "electrodes.myo.level=[0.1, 0.2]"
  myogen view view.shard=runs/run1/shard_003.h5 view.trial=s0007_x01_t00
"""
import argparse
import logging
import sys

from . import log as logsetup
from .commands import BANK_ACTIONS, COMMANDS, run
from .config import parse_override

TAKES_ARGS = ("biomime-bank", "download", "init")       # commands with positional arguments

log = logging.getLogger("myogen")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["simulate"]:
        return simulate_main(argv[1:])
    ap = argparse.ArgumentParser(prog="myogen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("simulate", *COMMANDS), metavar="command")
    ap.add_argument("args", nargs="*", metavar="KEY=VALUE",
                    help="config overrides (biomime-bank: ACTION first; download: what; init: optional DIR)")
    ap.add_argument("-c", "--config", default=None, help="config file ('default': the packaged defaults)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output (e.g. every finished trial)")
    ap.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    a = ap.parse_args(argv)
    logsetup.setup(_verbosity(a))
    positional = [x for x in a.args if "=" not in x]
    if a.command == "biomime-bank" and (len(positional) != 1 or positional[0] not in BANK_ACTIONS):
        ap.error(f"biomime-bank needs one action: {' | '.join(BANK_ACTIONS)}")
    if a.command == "init" and len(positional) > 1:
        ap.error("init takes at most one directory")
    if positional and a.command not in TAKES_ARGS:
        ap.error(f"unexpected arguments {positional} (overrides are KEY=VALUE)")
    _guarded(lambda: run(a.command, a.config, dict(parse_override(x) for x in a.args if "=" in x), positional))


def simulate_main(argv):
    """myogen simulate MOTION OUT [--config sim.yml] [--engine ...] [--device ...] [--seed N]"""
    from .emg.acquisition import DEVICES
    from .simulation.config import ENGINES, SimConfig
    ap = argparse.ArgumentParser(prog="myogen simulate", description="Simulate the sEMG of one movement recording "
                                 "(input format: myogen.motion.recording; output format: "
                                 "myogen.simulation.simulator; config: --write-config).")
    ap.add_argument("motion", nargs="?", help="movement recording (.h5 / .hdf5 / .npz)")
    ap.add_argument("out", nargs="?", help="output HDF5 file")
    ap.add_argument("-c", "--config", help="simulation config (YAML); default: the built-in defaults")
    ap.add_argument("--engine", choices=ENGINES, help="override engine.name")
    ap.add_argument("--device", choices=DEVICES, help="override device.name")
    ap.add_argument("--seed", type=int, help="override seed")
    ap.add_argument("--write-config", metavar="PATH", help="write the commented default config to PATH and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    logsetup.setup(_verbosity(a))
    if a.write_config:
        return _guarded(lambda: log.info(f"wrote {SimConfig.write_default(a.write_config)}"))
    if not (a.motion and a.out):
        ap.error("MOTION and OUT are required")

    def simulate():
        from .simulation.simulator import simulate
        cfg = SimConfig.load(a.config) if a.config else SimConfig()
        if a.engine:
            cfg.engine.name = a.engine
        if a.device:
            cfg.device.name = a.device
        if a.seed is not None:
            cfg.seed = a.seed
        simulate(a.motion, cfg, out=a.out)

    _guarded(simulate)


def _verbosity(a):
    return 1 if a.verbose else -1 if a.quiet else 0


def _guarded(fn):
    """Run fn; report expected errors as one log line and exit with a status instead of a traceback."""
    try:
        return fn()
    except KeyboardInterrupt:
        log.warning("interrupted")
        sys.exit(130)
    except (KeyError, ValueError, FileNotFoundError, FileExistsError, ImportError) as e:
        log.error(e.args[0] if e.args else str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
