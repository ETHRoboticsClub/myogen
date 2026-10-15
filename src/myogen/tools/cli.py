"""myogen: hand and wrist joint angles in, forearm sEMG out.

  myogen simulate MOTION OUT [--electrodes myo|otto_bock|ring|FILE.yml] [--subject FILE.yml]
  myogen replay RECORDING                  web viewer: the hand + the EMG
  myogen electrodes PRESET FILE.yml        write a preset as an editable electrode file
  myogen example FILE.h5                   write an example motion (open-close grasps)
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="myogen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    cmd = ap.add_subparsers(dest="command", required=True, metavar="command")

    p = cmd.add_parser("simulate", help="simulate the sEMG of a motion file")
    p.add_argument("motion", help="joint angle file (.h5 / .npz; format: see the README)")
    p.add_argument("out", help="output recording (.h5)")
    p.add_argument("--electrodes", default="myo", help="preset (ring | myo | otto_bock) or electrode YAML file")
    p.add_argument("--subject", help="subject YAML file (keys: myogen.Subject); default: the average subject")

    p = cmd.add_parser("replay", help="open a recording in the web viewer")
    p.add_argument("recording")
    p.add_argument("--port", type=int, default=8080)

    p = cmd.add_parser("electrodes", help="write an electrode preset to an editable YAML file")
    p.add_argument("preset", choices=("ring", "myo", "otto_bock"))
    p.add_argument("out")

    p = cmd.add_parser("example", help="write an example motion file")
    p.add_argument("out")
    p.add_argument("--duration", type=float, default=6.0)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(message)s")
    try:
        COMMANDS[a.command](a)
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as e:
        logging.getLogger("myogen").error(e)
        sys.exit(2)


def simulate(a: argparse.Namespace) -> None:
    import yaml

    from ..inputs.electrodes import Electrodes
    from ..inputs.subject import Subject
    from ..simulator import Simulator
    subject = Subject.from_dict(yaml.safe_load(open(a.subject))) if a.subject else None
    sim = Simulator(Electrodes.load(a.electrodes), subject=subject)
    print(f"wrote {sim.run(a.motion, progress=not a.quiet).save(a.out)}")


def replay(a: argparse.Namespace) -> None:
    from .viewer import replay
    replay(a.recording, port=a.port)


def electrodes(a: argparse.Namespace) -> None:
    from ..inputs.electrodes import Electrodes
    print(f"wrote {Electrodes.preset(a.preset).save(a.out)}")


def example(a: argparse.Namespace) -> None:
    from ..inputs.motion import example_motion
    print(f"wrote {example_motion(a.duration).save(a.out)}")


COMMANDS: dict[str, Callable[[argparse.Namespace], None]] = {
    "simulate": simulate, "replay": replay, "electrodes": electrodes, "example": example}

if __name__ == "__main__":
    main()
