"""Python entry point: the same commands as the CLI, for scripts and other repositories.

    import myogen as mg
    cfg = mg.load_config("my.yml", overrides={"generate.subjects": 4, "electrodes.myo.level": [0.1, 0.2]})
    records = mg.run("generate", cfg)                  # or mg.run("generate", overrides={...})
    mg.run("inspect", cfg, overrides={"inspect.trial": records[0]["id"]})
"""

import copy
import logging

from .config import INIT_TEMPLATE, LOCAL_CONFIG, apply_overrides, default_shard, load_config, resolve
from .log import console
from .log import setup as setup_logging

COMMANDS = ("init", "config", "download", "build-model", "ninapro-bank", "biomime-bank", "generate", "realism",
            "view", "inspect", "inspect-biomime", "inspect-muaps")
BANK_ACTIONS = ("build", "info", "regrid")


def run(command, config=None, overrides=None, args=()):
    """command: one of COMMANDS; config: a loaded config dict, a file path or None (see config.find_config);
    overrides: {'section.key': value}; args: positional arguments (biomime-bank: build | info | regrid;
    download: asset names or all; init: directory)."""
    if not logging.getLogger().handlers:
        setup_logging()
    if command not in COMMANDS:
        raise ValueError(f"unknown command {command!r}; one of {', '.join(COMMANDS)}")
    if command == "init":
        return _init(args[0] if args else None)
    cfg = config if isinstance(config, dict) else load_config(config)
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    if command == "config":
        import yaml
        from rich.syntax import Syntax
        shown = _relative_paths(copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")}))
        text = f"# {_rel(cfg['_file'])}\n" + yaml.safe_dump(shown, sort_keys=False, width=1000)
        if console.is_terminal:
            console.print(Syntax(text, "yaml", background_color="default", word_wrap=True))
        else:
            print(text, end="")
        return cfg
    if command == "download":
        from .download import run as download
        return download(cfg, list(args))
    if command == "build-model":
        from .build_model import main as build
        return build()
    if command == "ninapro-bank":
        from .ninapro import build_bank
        return build_bank(resolve(cfg, cfg["paths"]["latent_bank"]), resolve(cfg, cfg["paths"]["ninapro"]))
    if command == "biomime-bank":
        from .biomime import bank
        b, path = cfg["biomime_bank"], resolve(cfg, cfg["paths"]["muap_bank"])
        action = args[0] if args else None
        if action == "build":
            return bank.build(path, b["per_muscle"], b["seed"], b["threads"] or None,
                              weights=resolve(cfg, cfg["paths"]["biomime_weights"]))
        if action == "regrid":
            return bank.regrid(path, b["z0"], b["z1"])
        if action == "info":
            return bank.info(path)
        raise ValueError(f"biomime-bank needs an action: {' | '.join(BANK_ACTIONS)}")
    if command == "generate":
        from .generate import run as generate
        return generate(cfg)
    if command == "realism":
        from .realism import run as realism
        r = cfg["realism"]
        sim = resolve(cfg, r["sim"] or cfg["generate"]["out"])
        return realism(r["real"], str(sim), resolve(cfg, r["out"]), r["max_subjects"] or None,
                       bank=resolve(cfg, cfg["paths"]["latent_bank"]), root=resolve(cfg, cfg["paths"]["ninapro"]))
    if command == "view":
        from .sim import run as view
        v = cfg["view"]
        shard = default_shard(cfg, v["shard"]) if v["mode"] == "replay" else None
        return view(v["mode"], shard, v["trial"], v["emg_window"], v["port"])
    i = cfg["inspect"]
    if command == "inspect-muaps":
        from .biomime.inspect import plot_muaps
        return plot_muaps(i["muscle"] or None, resolve(cfg, i["out"]), resolve(cfg, cfg["paths"]["muap_bank"]))
    shard, out = default_shard(cfg, i["shard"]), resolve(cfg, i["out"])
    if command == "inspect":
        from .inspect_trial import plot
        return plot(shard, i["trial"] or None, out, i["seconds"])
    from .biomime.inspect import plot_trial
    return plot_trial(str(shard), i["trial"] or None, out, i["seconds"])


def _rel(p):
    """Path relative to the current directory when it is below it (for display)."""
    from pathlib import Path
    p = Path(p)
    return str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p)


def _relative_paths(cfg):
    from .config import PATH_KEYS
    for section, key in PATH_KEYS:
        if cfg.get(section, {}).get(key):
            cfg[section][key] = _rel(cfg[section][key])
    return cfg


def _init(path=None):
    """Write a starter myogen.yml (extends the defaults) into the current directory or `path`."""
    from pathlib import Path
    target = Path(path) if path else Path.cwd() / LOCAL_CONFIG
    if target.is_dir():
        target = target / LOCAL_CONFIG
    if target.exists():
        raise FileExistsError(f"{target} exists")
    target.write_text(INIT_TEMPLATE)
    console.print(f"wrote [bold]{target}[/]: commands run in {target.parent} now use it")
    return target


__all__ = ["COMMANDS", "load_config", "run"]
