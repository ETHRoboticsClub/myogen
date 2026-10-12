"""Config loading: extends, file-relative paths, overrides, unknown keys."""
from pathlib import Path

import pytest

from myogen.config import DEFAULT_CONFIG, load_config, parse_override, user_data_dir


def test_default_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MYOGEN_DATA", raising=False)
    cfg = load_config("default")
    assert Path(cfg["generate"]["out"]) == tmp_path / "runs" / "run1"          # outputs: where the command runs
    assert Path(cfg["paths"]["latent_bank"]) == user_data_dir() / "real" / "latent_bank.h5"
    monkeypatch.setenv("MYOGEN_DATA", str(tmp_path / "d"))
    assert Path(load_config("default")["paths"]["muap_bank"]) == tmp_path / "d" / "biomime" / "muap_bank.h5"


def test_data_dir_in_a_file(tmp_path):
    (tmp_path / "c.yml").write_text("extends: default\npaths:\n  data: inputs\n")
    cfg = load_config(tmp_path / "c.yml")
    assert Path(cfg["paths"]["ninapro"]) == tmp_path / "inputs" / "real"


def test_variant_extends_and_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "v.yml").write_text("extends: default\ngenerate:\n  out: ../out\n  device: myo\n")
    cfg = load_config(tmp_path / "sub" / "v.yml")
    assert Path(cfg["generate"]["out"]) == (tmp_path / "out").resolve()
    assert cfg["generate"]["device"] == "myo"
    assert cfg["generate"]["subjects"] == load_config(DEFAULT_CONFIG)["generate"]["subjects"]   # inherited


def test_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("default", dict([parse_override("generate.subjects=3"),
                                       parse_override("electrodes.myo.level=[0.1, 0.2]"),
                                       parse_override("generate.out=here")]))
    assert cfg["generate"]["subjects"] == 3
    assert cfg["electrodes"]["myo"]["level"] == [0.1, 0.2]
    assert Path(cfg["generate"]["out"]) == (tmp_path / "here").resolve()      # overrides: relative to cwd
    cfg = load_config("default", {"electrodes.myo.targets": ["FDS3_r"]})    # optional electrode key
    assert cfg["electrodes"]["myo"]["targets"] == ["FDS3_r"]


def test_unknown_override_is_an_error():
    with pytest.raises(KeyError):
        load_config("default", {"generate.subjetcs": 3})
    with pytest.raises(KeyError):
        load_config("default", {"electrodes.myoo.level": 0.1})


def test_unknown_file_key_warns(tmp_path, caplog):
    (tmp_path / "v.yml").write_text("extends: default\ngenerate:\n  subjetcs: 3\n")
    load_config(tmp_path / "v.yml")
    assert "generate.subjetcs" in caplog.text
