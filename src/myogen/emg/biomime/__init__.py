"""BioMime motor-unit-level sEMG engine.

BioMime (Ma et al. 2023, GPL-3.0, network vendored in vendor/) maps six motor-unit parameters to a MUAP on a fixed
10 x 32 electrode grid around one modelled forearm; NeuroMotion (Ma et al. 2024) couples it with a
musculoskeletal model. This package does the same on our MuJoCo model:

  generator.py   pretrained BioMime network: physical conditions in, MUAPs out
  anatomy.py     our muscles -> BioMime territories, motor-unit counts and sizes, grid placement
  bank.py        offline MUAP bank (motor units x fibre-length bins), `myogen biomime-bank`
  electrodes.py  any electrode layout as an interpolation of the grid, skin / fat filter
  mupool.py      motor-unit pools (Fuglevand rate model, LIF): drive -> spikes
  synth.py       BioMimeEMG: spikes x dynamic MUAPs for one subject / session

Setup: `myogen download muap-bank` (or `myogen download biomime-weights` + `myogen biomime-bank build`).
"""
