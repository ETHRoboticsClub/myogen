#!/usr/bin/env bash
# Set up the BioMime motor-unit-level EMG backend (third-party, GPL-3.0; kept out of this repository).
#   1. clone BioMime + NeuroMotion into third_party/ (pinned commits)
#   2. install BioMime and CPU PyTorch into .venv
#   3. download the pretrained generator from the authors' Google Drive
#   4. build the MUAP bank (python -m src.biomime.bank build)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p third_party/biomime_ckp
clone() {  # repo commit
  [ -d "third_party/$1" ] || git clone -q https://github.com/shihan-ma/$1.git "third_party/$1"
  git -C "third_party/$1" checkout -q "$2"
}
clone BioMime 3cdecc1
clone NeuroMotion 590369e
$PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install -q gdown easydict pyyaml ./third_party/BioMime
# Pretrained generator (BioMime README: https://drive.google.com/drive/folders/17Z2QH5NNaIv9p4iDq8HqytFaYk9Qnv2C).
# Only the state dict is used; it is loaded with torch.load(weights_only=True).
if [ ! -f third_party/biomime_ckp/model_linear.pth ]; then
  .venv/bin/gdown -q --folder https://drive.google.com/drive/folders/17Z2QH5NNaIv9p4iDq8HqytFaYk9Qnv2C -O third_party/biomime_ckp
  rm -rf third_party/biomime_ckp/ARMS_Wrist_Hand_Model_4.3.zip third_party/biomime_ckp/wrist_flex_ext_sub0_trial0 \
         third_party/biomime_ckp/muaps.npy                       # OpenSim model / pickled examples: not needed
fi
[ -f data/biomime/muap_bank.h5 ] || $PY -m src.biomime.bank build --per-muscle 128
echo "BioMime backend ready: python -m src.generate --emg-model biomime ..."
