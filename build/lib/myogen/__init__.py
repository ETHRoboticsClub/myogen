"""myogen: musculoskeletal forearm simulation and synthetic sEMG data generation.

    import myogen as mg
    mg.run("generate", "my.yml", overrides={"generate.subjects": 4})

See api.py for the Python entry point and cli.py / `myogen --help` for the command line.
"""
from .api import COMMANDS, run
from .config import load_config

__all__ = ["COMMANDS", "load_config", "run"]
