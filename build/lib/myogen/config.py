"""Central configuration: one YAML file per run, plus overrides from the command line or Python.

Which file: `--config <file>` (`default` for the packaged defaults) > $MYOGEN_CONFIG > ./myogen.yml (see
`myogen init`) > the packaged defaults (defaults.yml next to this module).

A file can start with `extends: <other.yml>` (or `extends: default`) and only list what it changes: tables are
merged key by key, everything else (numbers, strings, lists) replaces the base value. Keys that the defaults do
not have are reported, so typos do not pass silently.

Paths: relative paths are relative to the file that sets them; in the packaged defaults and in overrides, to the
current directory. `{data}` stands for paths.data, the data directory for downloaded / built inputs (NinaPro,
latent bank, BioMime weights and MUAP bank): by default $MYOGEN_DATA or the per-user data directory
(~/.local/share/myogen on Linux), see `myogen download`.
"""
import copy
import logging
import os
import sys
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE / "defaults.yml"
LOCAL_CONFIG = "myogen.yml"
# Values that are file system paths ("" = unset / use the default behaviour).
PATH_KEYS = {("paths", "data"), ("paths", "ninapro"), ("paths", "latent_bank"), ("paths", "biomime_weights"),
             ("paths", "muap_bank"), ("generate", "out"), ("view", "shard"), ("inspect", "shard"),
             ("inspect", "out"), ("realism", "sim"), ("realism", "out")}
# Optional per-device electrode keys (not every device block in the default config has them).
OPTIONAL_ELECTRODE_KEYS = {"targets", "target_level", "level_amputee", "amputee_margin"}

log = logging.getLogger(__name__)


def user_data_dir():
    """$MYOGEN_DATA, or the per-user data directory of this OS."""
    if os.environ.get("MYOGEN_DATA"):
        return Path(os.environ["MYOGEN_DATA"]).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "myogen"


def _absolutise(cfg, base_dir):
    for section, key in PATH_KEYS:
        v = cfg[section].get(key) if isinstance(cfg.get(section), dict) else None
        if isinstance(v, str) and v and not v.startswith("{data}") and not Path(v).expanduser().is_absolute():
            cfg[section][key] = str((base_dir / v).resolve())
    return cfg


def _expand_data(cfg):
    """Resolve paths.data and the `{data}` placeholder in every path."""
    data = cfg["paths"].get("data") or str(user_data_dir())
    cfg["paths"]["data"] = data
    for section, key in PATH_KEYS:
        v = cfg.get(section, {}).get(key)
        if isinstance(v, str) and v:
            cfg[section][key] = str(Path(v.replace("{data}", data)).expanduser())
    return cfg


def _optional(path):
    return len(path) == 3 and path[0] == "electrodes" and path[2] in OPTIONAL_ELECTRODE_KEYS


def _merge(base, over, path=()):
    out = dict(base)
    for k, v in over.items():
        here = path + (k,)
        if k not in base and not _optional(here):
            log.warning("unknown config key [bold]%s[/] (not in the default config)", ".".join(map(str, here)))
        out[k] = _merge(out[k], v, here) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path):
    """(config, whether it extends another file)."""
    with open(path) as f:
        # The packaged defaults' relative paths (outputs) are relative to where the command runs.
        cfg = _absolutise(yaml.safe_load(f) or {}, Path.cwd() if path == DEFAULT_CONFIG else path.parent)
    base = cfg.pop("extends", None)
    if base is None:
        return cfg, False
    base_path = DEFAULT_CONFIG if base == "default" else (path.parent / base).resolve()
    return _merge(_read(base_path)[0], cfg), True


def find_config(path=None):
    """The config file to use (see module docstring)."""
    if path == "default":
        return DEFAULT_CONFIG
    if path or os.environ.get("MYOGEN_CONFIG"):
        return Path(path or os.environ["MYOGEN_CONFIG"]).expanduser().resolve()
    local = Path.cwd() / LOCAL_CONFIG
    return local.resolve() if local.exists() else DEFAULT_CONFIG


def parse_override(text):
    """'generate.subjects=20' -> ('generate.subjects', 20); values are YAML ([0.1, 0.2], true, myo, ...)."""
    if "=" not in text:
        raise ValueError(f"override {text!r} is not KEY=VALUE")
    key, value = text.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def apply_overrides(cfg, overrides):
    """overrides: {'generate.subjects': 20, ...} (or (key, value) pairs). Unknown keys are errors; relative
    paths are relative to the current directory."""
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides.items() if isinstance(overrides, dict) else overrides):
        key = tuple(key.split("."))
        node = cfg
        for i, k in enumerate(key[:-1]):
            if not isinstance(node.get(k), dict):
                raise KeyError(f"unknown config key {'.'.join(key[:i + 1])!r}")
            node = node[k]
        if key[-1] not in node and not _optional(key):
            raise KeyError(f"unknown config key {'.'.join(key)!r}")
        if key in PATH_KEYS and isinstance(value, str) and value and not value.startswith("{data}"):
            value = str(Path(value).expanduser().resolve())
        node[key[-1]] = value
    return cfg


def load_config(path=None, overrides=None):
    """Resolved configuration dict: file (see find_config) + overrides (see apply_overrides)."""
    path = find_config(path)
    cfg, extends = _read(path)
    if not extends and path != DEFAULT_CONFIG:
        # A standalone file: check its keys against the default config.
        _merge(_read(DEFAULT_CONFIG)[0], cfg)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg = _expand_data(cfg)
    cfg["_file"] = str(path)
    return cfg


def resolve(cfg, p):
    """A path from the config (absolute after loading; relative ones are taken from the current directory)."""
    return Path(p).resolve()


def default_shard(cfg, shard=None):
    """The configured shard, or the first shard of generate.out."""
    if shard:
        return resolve(cfg, shard)
    shards = sorted(resolve(cfg, cfg["generate"]["out"]).glob("shard_*.h5"))
    if not shards:
        raise FileNotFoundError(f"no shard set and no shards in {cfg['generate']['out']}: run `myogen generate`")
    return shards[0]


INIT_TEMPLATE = """\
# myogen configuration for this folder: `myogen <command>` run here picks it up automatically.
# It extends the default configuration, so list only what you change (all keys: myogen config --config default).
# Relative paths are relative to this file. Any value can also be overridden for one run:
#   myogen generate generate.subjects=4 "electrodes.myo.level=[0.1, 0.2]"
extends: default

generate:
  out: data/run1
  device: myo
  subjects: 8

# electrodes:
#   myo:
#     level: [0.06, 0.18]          # ring position, fraction of forearm length from the elbow
#     rotation_deg: [-180, 180]    # armband rotation (0 ~ radial border, +90 dorsal, -90 palmar)
"""
