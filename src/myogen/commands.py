"""The `myogen` dataset commands, callable from Python with the same config and overrides as the CLI.

    import myogen as mg
    cfg = mg.load_config("my.yml", overrides={"generate.subjects": 4})
    records = mg.run("generate", cfg)
    mg.run("inspect", cfg, overrides={"inspect.trial": records[0]["id"]})
"""
import copy
import logging
from pathlib import Path

from .config import INIT_TEMPLATE, LOCAL_CONFIG, PATH_KEYS, apply_overrides, as_path, default_shard, load_config
from .log import console
from .log import setup as setup_logging

BANK_ACTIONS = ("build", "info", "regrid")


def run(command, config=None, overrides=None, args=()):
    """command: one of COMMANDS; config: a loaded config dict, a file path or None (see config.find_config);
    overrides: {'section.key': value}; args: positional arguments (init: directory; download: asset names;
    biomime-bank: build | info | regrid)."""
    if not logging.getLogger().handlers:
        setup_logging()
    if command not in COMMANDS:
        raise ValueError(f"unknown command {command!r}; one of {', '.join(COMMANDS)}")
    if command == "init":
        return init(args[0] if args else None)
    cfg = config if isinstance(config, dict) else load_config(config)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return COMMANDS[command](cfg, list(args))


def init(path=None):
    """Write a starter myogen.yml (extending the defaults) into `path` or the current directory."""
    target = Path(path) if path else Path.cwd() / LOCAL_CONFIG
    if target.is_dir():
        target = target / LOCAL_CONFIG
    if target.exists():
        raise FileExistsError(f"{target} exists")
    target.write_text(INIT_TEMPLATE)
    console.print(f"wrote [bold]{target}[/]: commands run in {target.parent} now use it")
    return target


def show_config(cfg, args):
    import yaml
    from rich.syntax import Syntax
    shown = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    for section, key in PATH_KEYS:
        if shown.get(section, {}).get(key):
            shown[section][key] = _relative(shown[section][key])
    text = f"# {_relative(cfg['_file'])}\n" + yaml.safe_dump(shown, sort_keys=False, width=1000)
    if console.is_terminal:
        console.print(Syntax(text, "yaml", background_color="default", word_wrap=True))
    else:
        print(text, end="")
    return cfg


def download(cfg, args):
    from .dataset.download import run
    return run(cfg, args)


def build_model(cfg, args):
    from .body.build import build
    return build()


def ninapro_bank(cfg, args):
    from .motion.ninapro import build_bank
    return build_bank(as_path(cfg["paths"]["latent_bank"]), as_path(cfg["paths"]["ninapro"]))


def biomime_bank(cfg, args):
    from .emg.biomime import bank
    b, path = cfg["biomime_bank"], as_path(cfg["paths"]["muap_bank"])
    action = args[0] if args else None
    if action == "build":
        return bank.build(path, b["per_muscle"], b["seed"], b["threads"] or None,
                          weights=as_path(cfg["paths"]["biomime_weights"]))
    if action == "regrid":
        return bank.regrid(path, b["z0"], b["z1"])
    if action == "info":
        return bank.info(path)
    raise ValueError(f"biomime-bank needs an action: {' | '.join(BANK_ACTIONS)}")


def generate(cfg, args):
    from .dataset.generate import run
    return run(cfg)


def realism(cfg, args):
    from .dataset.realism import run
    r = cfg["realism"]
    return run(r["real"], str(as_path(r["sim"] or cfg["generate"]["out"])), as_path(r["out"]),
               r["max_subjects"] or None, bank=as_path(cfg["paths"]["latent_bank"]),
               root=as_path(cfg["paths"]["ninapro"]))


def view(cfg, args):
    from .viewer.view import run
    v = cfg["view"]
    shard = default_shard(cfg, v["shard"]) if v["mode"] == "replay" else None
    return run(v["mode"], shard, v["trial"], v["emg_window"], v["port"])


def inspect(cfg, args):
    from .viewer.plots import plot_trial
    i = cfg["inspect"]
    return plot_trial(default_shard(cfg, i["shard"]), i["trial"] or None, as_path(i["out"]), i["seconds"])


def inspect_biomime(cfg, args):
    from .viewer.plots import plot_spikes
    i = cfg["inspect"]
    return plot_spikes(str(default_shard(cfg, i["shard"])), i["trial"] or None, as_path(i["out"]), i["seconds"])


def inspect_muaps(cfg, args):
    from .viewer.plots import plot_muaps
    i = cfg["inspect"]
    return plot_muaps(i["muscle"] or None, as_path(i["out"]), as_path(cfg["paths"]["muap_bank"]))


COMMANDS = {
    "init": init, "config": show_config, "download": download, "build-model": build_model,
    "ninapro-bank": ninapro_bank, "biomime-bank": biomime_bank, "generate": generate, "realism": realism,
    "view": view, "inspect": inspect, "inspect-biomime": inspect_biomime, "inspect-muaps": inspect_muaps,
}


def _relative(p):
    """p relative to the current directory when it is below it (for display)."""
    p = Path(p)
    return str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p)
