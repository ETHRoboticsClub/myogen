"""`myogen download`: inputs too large for the package, into paths.* (default: the user data directory).

  latent-bank      real NinaPro motion (generate.motion: ninapro / mix)     from download.base_url
  muap-bank        BioMime MUAP bank (generate.emg_model: biomime)          from download.base_url
  biomime-weights  BioMime generator, only to rebuild the MUAP bank         from the authors' Google Drive

Files are checked against the SHA-256 this code was validated with; a mismatch is an error.
"""
import hashlib
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from ..log import console

log = logging.getLogger(__name__)

# name -> (config path key, file name on download.base_url (None: other source), sha256, size in bytes)
ASSETS = {
    "latent-bank": ("latent_bank", "latent_bank.h5",
                    "1ed2a88487cfaa37237f6e3d6bc41630e187a2e9032462b853e4858b12a59e0c", 131_086_815),
    "muap-bank": ("muap_bank", "muap_bank.h5",
                  "5bf6d1b8ebdc707b21647f5d3f4d1d0258e02d48099aaab39edd26e22158655e", 829_865_160),
    "biomime-weights": ("biomime_weights", None,
                        "4042113a7a90d2fa30b8e6df304542733091221d494dac564e5b09a04556484f", 57_560_093),
}
# BioMime README: https://github.com/shihan-ma/BioMime (pretrained generator, linear KL annealing)
BIOMIME_DRIVE_FOLDER = "https://drive.google.com/drive/folders/17Z2QH5NNaIv9p4iDq8HqytFaYk9Qnv2C"


def run(cfg, names):
    """names: asset names (see ASSETS) or ["all"]."""
    names = list(ASSETS) if not names or "all" in names else names
    for name in names:
        if name not in ASSETS:
            raise ValueError(f"unknown download {name!r}; one of {', '.join(ASSETS)}, all")
        key, filename, sha, size = ASSETS[name]
        dest = Path(cfg["paths"][key])
        if dest.exists():
            log.info(f"{name}: {dest} exists (delete it to download again)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if filename is None:
            _biomime_weights(dest)
        else:
            base = cfg["download"]["base_url"]
            if not base:
                raise ValueError(f"download.base_url is not set, so {name} cannot be downloaded; set it to where "
                                 f"{filename} is hosted, or build it locally ({_local_alternative(name)})")
            _fetch(f"{base.rstrip('/')}/{filename}", dest, size)
        _verify(dest, sha, name)
        log.info(f"{name}: [green]ok[/] -> {dest}")


def _local_alternative(name):
    return {"latent-bank": "`myogen ninapro-bank` with raw NinaPro data in paths.ninapro",
            "muap-bank": "`myogen download biomime-weights` then `myogen biomime-bank build`"}.get(name, "")


def _fetch(url, dest, size):
    tmp = dest.with_suffix(dest.suffix + ".part")
    columns = (TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
               TimeRemainingColumn())
    with Progress(*columns, console=console) as progress, urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        task = progress.add_task(dest.name, total=int(r.headers.get("Content-Length") or size))
        while chunk := r.read(1 << 20):
            f.write(chunk)
            progress.update(task, advance=len(chunk))
    tmp.replace(dest)


def _biomime_weights(dest):
    try:
        import gdown
    except ImportError as e:
        raise ImportError("the BioMime weights download needs gdown: pip install 'myogen\\[biomime]'") from e
    with tempfile.TemporaryDirectory() as tmp:
        log.info("downloading the BioMime checkpoint folder from the authors' Google Drive ...")
        # The folder also holds an OpenSim model and pickled examples; only the state dict is kept (it is loaded
        # with torch.load(weights_only=True)).
        gdown.download_folder(BIOMIME_DRIVE_FOLDER, output=tmp, quiet=True)
        shutil.move(str(Path(tmp) / "model_linear.pth"), dest)


def _verify(path, sha, name):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    if h.hexdigest() != sha:
        bad = path.with_suffix(path.suffix + ".bad")
        path.replace(bad)
        raise ValueError(f"{name}: checksum mismatch (got {h.hexdigest()[:12]}..., expected {sha[:12]}...); "
                         f"moved to {bad}")
