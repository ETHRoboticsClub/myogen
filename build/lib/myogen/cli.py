"""myogen: synthetic forearm sEMG data generation. All parameters live in a YAML config (see config.py).

  command              what                                                config
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
from .api import BANK_ACTIONS, COMMANDS, run
from .config import parse_override

log = logging.getLogger("myogen")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="myogen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS, metavar="command")
    ap.add_argument("args", nargs="*", metavar="KEY=VALUE",
                    help="config overrides (biomime-bank: ACTION first; download: what; init: optional DIR)")
    ap.add_argument("-c", "--config", default=None, help="config file ('default': the repository's config.yml)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug output (e.g. every finished trial)")
    ap.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    a = ap.parse_args(argv)
    logsetup.setup(1 if a.verbose else -1 if a.quiet else 0)

    positional = [x for x in a.args if "=" not in x]
    try:
        overrides = dict(parse_override(x) for x in a.args if "=" in x)
        if a.command == "biomime-bank" and (len(positional) != 1 or positional[0] not in BANK_ACTIONS):
            ap.error(f"biomime-bank needs one action: {' | '.join(BANK_ACTIONS)}")
        if a.command == "init" and len(positional) > 1:
            ap.error("init takes at most one directory")
        if positional and a.command not in ("biomime-bank", "download", "init"):
            ap.error(f"unexpected arguments {positional} (overrides are KEY=VALUE)")
        run(a.command, a.config, overrides, positional)
    except KeyboardInterrupt:
        log.warning("interrupted")
        sys.exit(130)
    except (KeyError, ValueError, FileNotFoundError, FileExistsError, ImportError) as e:
        log.error(e.args[0] if e.args else str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
