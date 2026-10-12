"""BioMime-based motor-unit-level sEMG for the myogen pipeline.

BioMime (Ma et al. 2023, https://github.com/shihan-ma/BioMime, GPL-3.0) is a conditional generative model that
maps six motor-unit parameters (fibre number, depth, angle, innervation zone, conduction velocity, fibre length)
to a monopolar MUAP on a fixed 10 x 32 electrode grid around one numerically modelled forearm. NeuroMotion
(Ma et al. 2024, PLOS Comput Biol) couples it with a musculoskeletal model and motor-unit pools. This package
reimplements that idea on top of our MuJoCo model, with BioMime's generator network vendored in vendor/
(GPL-3.0, provenance and citations in vendor/README.md), and adds what the pipeline needs for data generation:

  model.py      load the pretrained generator, condition normalisation
  anatomy.py    map our 48 MuJoCo muscles into BioMime territory coordinates (anchors + fitted transform),
                motor-unit counts and sizes per muscle, grid geometry assumptions
  bank.py       offline MUAP bank: per muscle, motor units x fibre-length bins on the full grid (HDF5)
  electrodes.py arbitrary electrode layouts (rings, targeted, HD) as interpolation of the grid, bipolar
                derivation, skin/fat spatial filtering
  mupool.py     motor-unit pools (Fuglevand rate model, LIF) behind one interface; spikes for SNN work later
  synth.py      per subject/session EMG synthesis: spikes x dynamic MUAPs via low-rank convolution

Setup: scripts/setup_biomime.sh (installs CPU PyTorch, downloads the pretrained weights to data/biomime/ and
builds the MUAP bank with `myogen biomime-bank build`).
"""
