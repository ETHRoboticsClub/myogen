"""The BioMime engine (Ma et al. 2023, network vendored in vendor/, GPL-3.0).

  engine.py    BioMimeEngine: electrodes on the grid, motor units, spikes x MUAPs
  network.py   the pretrained network: physical motor-unit conditions in, MUAPs out
  anatomy.py   our muscles -> BioMime territories, motor-unit counts and sizes, grid placement
  bank.py      the MUAP bank, built once from the published weights (`myogen biomime-setup`)
  pools.py     motor-unit pools (Fuglevand rate model, LIF): drive -> spikes
"""
from .engine import BioMimeEngine

__all__ = ["BioMimeEngine"]
