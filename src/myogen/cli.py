"""myogen: hand and wrist joint angles in, forearm sEMG out.

  myogen simulate MOTION OUT [--engine analytic|biomime] [--electrodes myo|otto_bock|ring|FILE.yml]
  myogen replay RECORDING                  web viewer: the hand + the EMG
  myogen electrodes PRESET FILE.yml        write a preset as an editable electrode file
  myogen example FILE.h5                   write an example motion (open-close grasps)
  myogen biomime-setup                     download the BioMime weights and build the MUAP bank (once)
"""
import argparse
import logging
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="myogen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    cmd = ap.add_subparsers(dest="command", required=True, metavar="command")

    p = cmd.add_parser("simulate", help="simulate the sEMG of a motion file")
    p.add_argument("motion", help="joint angle file (.h5 / .npz; format: python -m pydoc myogen.motion)")
    p.add_argument("out", help="output recording (.h5)")
    p.add_argument("--engine", choices=("analytic", "biomime"), default="analytic")
    p.add_argument("--electrodes", default="myo", help="preset (ring | myo | otto_bock) or electrode YAML file")
    p.add_argument("--subject", help="subject YAML file (keys: myogen.Subject); default: the average subject")
    p.add_argument("--muap-bank", help="biomime: MUAP bank path (default: where biomime-setup puts it)")
    p.add_argument("--gain", type=float, default=1.0, help="biomime: global amplitude (1 = physical)")
    p.add_argument("--save-spikes", action="store_true", help="biomime: store the motor-unit spike trains")

    p = cmd.add_parser("replay", help="open a recording in the web viewer")
    p.add_argument("recording")
    p.add_argument("--port", type=int, default=8080)

    p = cmd.add_parser("electrodes", help="write an electrode preset to an editable YAML file")
    p.add_argument("preset", choices=("ring", "myo", "otto_bock"))
    p.add_argument("out")

    p = cmd.add_parser("example", help="write an example motion file")
    p.add_argument("out")
    p.add_argument("--duration", type=float, default=6.0)

    p = cmd.add_parser("biomime-setup", help="download the BioMime weights and build the MUAP bank")
    p.add_argument("--bank", help="where to write the bank (default: $MYOGEN_DATA/biomime/muap_bank.h5)")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s")
    try:
        COMMANDS[a.command](a)
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as e:
        logging.getLogger("myogen").error(e)
        sys.exit(2)


def simulate(a):
    import yaml

    from .electrodes import Electrodes
    from .simulator import Simulator
    from .subject import Subject
    subject = Subject.from_dict(yaml.safe_load(open(a.subject))) if a.subject else None
    options = {"muap_bank": a.muap_bank, "gain": a.gain, "save_spikes": a.save_spikes} if a.engine == "biomime" else {}
    sim = Simulator(Electrodes.load(a.electrodes), engine=a.engine, subject=subject, **options)
    print(f"wrote {sim.run(a.motion).save(a.out)}")


def replay(a):
    from .viewer import replay
    replay(a.recording, port=a.port)


def electrodes(a):
    from .electrodes import Electrodes
    print(f"wrote {Electrodes.preset(a.preset).save(a.out)}")


def example(a):
    from .motion import example_motion
    print(f"wrote {example_motion(a.duration).save(a.out)}")


def biomime_setup(a):
    from .engines.biomime.bank import DEFAULT_BANK, setup
    print(f"wrote {setup(a.bank or DEFAULT_BANK)}")


COMMANDS = {"simulate": simulate, "replay": replay, "electrodes": electrodes, "example": example,
            "biomime-setup": biomime_setup}

if __name__ == "__main__":
    main()
