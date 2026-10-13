# Vendored: BioMime generator network

`generator.py` and `modules.py` are copied from **BioMime**
(https://github.com/shihan-ma/BioMime, commit `3cdecc1945b9e70aaafd66b42a901ca1febc5425`,
`BioMime/models/`), released under the **GNU General Public License v3.0** (full text in `LICENSE`).
`generator.yaml` is the `Model.Generator` section of that commit's `BioMime/configs/config.yaml`.

Only the generator is kept (inference); training code, the discriminator, data loaders and plotting are not.
The only modification is the import in `generator.py` (`from BioMime.models.modules import ...` →
`from .modules import ...`).

The pretrained weights (`model_linear.pth`) are not stored in the repository. `scripts/setup_biomime.sh`
downloads them from the authors' Google Drive into `data/biomime/`.

The motor-unit territory table in `../anatomy.py` (`ANCHOR_TABLE`) is taken from **NeuroMotion**
(https://github.com/shihan-ma/NeuroMotion, commit `590369ec2f395e6e228aa9dc58bf0fb87a2c0329`,
`NeuroMotion/MNPoollib/mn_params.py`, GPL-3.0). No NeuroMotion code is used.

## Cite

Both entries as given in the NeuroMotion paper (`docs/`, PLoS Comput Biol 20(7): e1012257). BioMime is cited
there as the arXiv preprint; check for a later journal version before publishing.

```bibtex
@article{ma2022biomime,
  title   = {Conditional generative models for simulation of {EMG} during naturalistic movements},
  author  = {Ma, Shihan and Clarke, Alexander Kenneth and Maksymenko, Kostiantyn and Deslauriers-Gauthier,
             Samuel and Sheng, Xinjun and Zhu, Xiangyang and Farina, Dario},
  journal = {arXiv preprint arXiv:2211.01856},
  year    = {2022}
}
@article{ma2024neuromotion,
  title   = {{NeuroMotion}: Open-source platform with neuromechanical and deep network modules to generate
             surface {EMG} signals during voluntary movement},
  author  = {Ma, Shihan and Mendez Guerra, Irene and Caillet, Arnault Hubert and Zhao, Jiamin and
             Clarke, Alexander Kenneth and Maksymenko, Kostiantyn and Deslauriers-Gauthier, Samuel and
             Sheng, Xinjun and Zhu, Xiangyang and Farina, Dario},
  journal = {PLoS Computational Biology},
  volume  = {20},
  number  = {7},
  pages   = {e1012257},
  year    = {2024},
  doi     = {10.1371/journal.pcbi.1012257}
}
```
