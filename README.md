# myogen — synthetic forearm sEMG generator on a musculoskeletal forearm model

A MuJoCo forearm/hand musculoskeletal model (**myoforearm**, built from MyoSim's **MyoArm**: 24 DoF, 48 muscles), a muscle-driven
controller that reproduces natural hand movement (synthetic or **real NinaPro glove trajectories**), and a sEMG
forward model that emulates the **NinaPro DB1 (Otto Bock)** and **DB5 (Myo)** recording hardware. The output is
physically consistent (intent, kinematics, muscle activity, multichannel EMG) data for pretraining the EMG → hand
model, plus a **realism report** that checks the synthetic EMG against the real datasets.

## Install

```bash
pip install "git+https://github.com/<user>/myogen.git"                 # the `myogen` command + Python package
pip install "myogen[biomime] @ git+https://github.com/<user>/myogen.git" \
    --extra-index-url https://download.pytorch.org/whl/cpu              # + BioMime backend (CPU PyTorch)
```

The package contains the musculoskeletal model and the default configuration, so it runs without a clone or
any download: `myogen generate` makes a dataset with synthetic motion and the analytic EMG model. Larger inputs
are fetched on demand into the data directory (`paths.data`: `$MYOGEN_DATA`, default `~/.local/share/myogen`):

```bash
myogen download latent-bank        # real NinaPro intent trajectories (generate.motion: ninapro | mix)
myogen download muap-bank          # BioMime MUAP bank (generate.emg_model: biomime)
myogen download biomime-weights    # BioMime pretrained network, only to rebuild the MUAP bank
```

`latent-bank` and `muap-bank` come from `download.base_url` (where the release files are hosted) and are
checked against pinned SHA-256 sums; the BioMime weights come from the authors' Google Drive. Instead of
downloading: `myogen ninapro-bank` builds the latent bank from raw NinaPro data in `paths.ninapro`, and
`myogen biomime-bank build` builds the MUAP bank from the weights (~10 min). `realism` needs the raw NinaPro
recordings in `paths.ninapro` (`ninapro_db1/s*/`, `ninapro_db5/s*/`).

**Development** (this repository): `python -m venv .venv && .venv/bin/pip install -e ".[test]"`. The root
`myogen.yml` points `paths.data` at the repository's `data/` and keeps the development defaults (mixed motion).

## Use

```bash
myogen generate          # dataset as configured (generate, electrodes.<device>), with a progress display
myogen view              # 3D replay at http://localhost:8080: arm, skin, electrodes, sources, live EMG
myogen inspect           # PNG of one trial
myogen realism           # compare with NinaPro
myogen --help            # all commands; -v shows every finished trial, -q only warnings
```

**Configuration.** Every parameter lives in a YAML config, so commands take no arguments. Which file is used:
`--config FILE` > `$MYOGEN_CONFIG` > `./myogen.yml` > the packaged defaults (`src/myogen/defaults.yml`, with
comments for every key; print them with `myogen config -c default`). Electrode placement is in `electrodes.<device>` (ring level along the forearm, rotation
in degrees, rings, jitter, targeted electrodes, inter-electrode distance); each value is a `[lo, hi]` range
sampled per subject or a single fixed number. `generate` copies the resolved config into `<out>/config.json`.

- **Variants** start with `extends: default` (or another file) and list only what they change; relative paths are
  relative to the file that sets them (in the packaged defaults and in overrides: to the current directory). `configs/` has the NinaPro validation sets:
  `myogen generate -c configs/val_db5.yml && myogen realism -c configs/val_db5.yml`.
- **Overrides** for one run: `KEY=VALUE` after the command (VALUE is YAML); unknown keys are errors.
  `myogen generate generate.subjects=4 "electrodes.myo.level=[0.1, 0.2]" generate.live_view=true`
- **From another project**: `myogen init` writes a `myogen.yml` there (extending the defaults); commands
  run in that folder use it and write their output next to it.
- **From Python**:
  ```python
  import myogen as mg
  cfg = mg.load_config("my.yml", overrides={"generate.subjects": 4, "electrodes.myo.rotation_deg": 90})
  records = mg.run("generate", cfg)            # manifest records; any CLI command works the same way
  ```
- `myogen config [-c FILE] [KEY=VALUE ...]` prints the resolved configuration.

**Live generation** (`generate.live_view: true`): while generating, worker 0 streams its simulation to the web
viewer (`view.port`): the arm as it is simulated (~2x real time) with the trial's recording setup, muscle sources
coloured by the controller's excitation and the trial / overall progress. EMG is synthesised at the end of each
trial, so it is not shown live; replay finished trials with `myogen view`. The viewer stays open after the
run until Ctrl+C.

**Viewer** (`view.mode: replay`, the default). It replays a generated trial (`view.shard` / `view.trial`, default
the first one in `generate.out`) with its recording setup: the subject's skin, the electrodes (coloured by channel
EMG activity, bipolar poles for BioMime trials), the muscle source points coloured by excitation, gravity (the
camera is oriented so it points down) and the armrest. The side panel shows the session parameters and a
scrolling plot of every EMG channel in the electrode colours. Other modes are `demo` (live random natural
motion) and `live` (muscle excitations over UDP, see `controller.py`).

Throughput on this machine (16 cores): ~1.9× real time per worker, **~13 h of data per wall-clock hour** with 14
workers.

## Layout

| Path | Content |
|---|---|
| `src/myogen/defaults.yml`, `myogen.yml`, `configs/` | packaged defaults; this repository's development config; variants for the NinaPro validation sets |
| `src/myogen/cli.py`, `api.py`, `config.py`, `log.py` | command line, Python entry point (`myogen.run`), config loading (`extends:`, overrides, paths), rich logging |
| `src/myogen/build_model.py` | builds the packaged `src/myogen/models/myoforearm_r.xml` from MyoArm (pruning, seated shoulder pose) |
| `src/myogen/model.py` | model loading (`myoforearm` default, `myohand` legacy), joint/muscle bookkeeping by name, belly geometry |
| `src/myogen/motion.py` | 10-D posture latent, prototypes, `JointMap`, synthetic trajectory generator |
| `src/myogen/ninapro.py` | DB1/DB5 loaders, CyberGlove → latent retargeting, `LatentBank` of real intent trajectories |
| `src/myogen/tracker.py` | computed muscle control with reserves, co-contraction, leaky smoothness, armrest support |
| `src/myogen/emg.py` | volume-conductor geometry, source synthesis, device emulation (raw / Otto Bock / Myo) |
| `src/myogen/subject.py` | every randomised subject / session / device parameter |
| `src/myogen/sim.py`, `overlay.py`, `live.py` | web viewer; recording-setup overlay and scrolling EMG; live view during generation |
| `src/myogen/generate.py` | parallel generation, passive equilibrium, QC, HDF5 shards |
| `src/myogen/realism.py` | realism gates against NinaPro |
| `src/myogen/biomime/` | BioMime motor-unit-level EMG backend (section 4); `vendor/` holds BioMime's generator network (GPL-3.0) |
| `tests/` | `pytest` checks for the config loader and the BioMime backend |
| `src/myogen/download.py` | `myogen download`: latent bank, MUAP bank, BioMime weights (checksummed) |
| `src/myogen/models/` | the pinned model snapshot + meshes (packaged, versioned with the code) |
| `data/real/ninapro_db1/` | 5 DB1 subjects copied from emg2action (raw `.mat`) |
| `data/real/ninapro_db5/` | all 10 DB5 subjects (`ninapro.hevs.ch/files/DB5_Preproc`; cite Pizzolato et al. 2017) |
| `reports/realism_db{1,5}/` | latest realism results (`realism.json`, `realism.png`) |

---

## 1. Model: why MyoArm-derived and not MyoHand

| | MyoHand (MyoSim) | **myoforearm** (this package) |
|---|---|---|
| DoF | 23 (wrist + hand) | 24 (+ elbow, posture variable) |
| Muscles | 39 | 48: + **SUP, BRD, ANC** (bellies in the proximal forearm, i.e. right under the DB1/DB5 rings), BIC×2, BRA, TRI×3 |
| Supination | no supinator/biceps → reserve torques | muscle-driven (reserve 0.09 → ≤ 0.015 Nm) |
| Shoulder pose | 90° elevation preview pose | **seated, arm by the side**: TRIlong passive force 200–530 N → ~0; passive equilibrium elbow 65°, forearm −19° (physiological) |

`src/myogen/build_model.py` documents the pruning. The shoulder pose was chosen by a grid search minimising passive
muscle force over the elbow range. The original pose made the elbow flexors fight a permanent 7–9 Nm passive
extension torque, i.e. tonic BRA/BRD EMG at rest. The thumb still lacks APB/FPB/AdP (intrinsic muscles,
invisible to forearm electrodes and absent after amputation) and keeps cheap reserves.

## 2. Pipeline

```
 subject / session / device sampler
        │
 [1] intent  z(t) ∈ R^10: synthetic generator  or  real NinaPro glove trajectory (LatentBank)
        │    JointMap → q_ref(t) (23 hand joints) + elbow posture
 [2] passive equilibrium: joint springs calibrated so the subject's relaxed posture is the passive equilibrium
 [3] computed muscle control @100 Hz: inverse dynamics → bounded LSQ over 48 activations + reserves
        │    effort, co-contraction c(t) (episodes × engagement + holding tone + stiffening while moving),
        │    leaky smoothness (τ ≈ 100 ms), armrest support of static torques, grip loads
 [4] forward physics → q, q̇, activation
 [5] EMG: belly sources per tick → exp(−d/decay) gains → band-limited sources × drive^β → device front-end
        │    raw (mV, any rate) | otto_bock (10 ch, rectified/smoothed, 100 Hz, V) | myo (16 ch, int8, 200 Hz)
 [6] QC + HDF5
```

### Motion sources (`generate.motion: synthetic | ninapro | mix`)

`src/myogen/ninapro.py` retargets the 22-sensor CyberGlove to the same 10-D latent the generator uses. The sensor order
was verified against the single-finger movements of exercise 1. Two channels the glove does not measure are
recovered and flagged:

- **pronation/supination**: in DB5, from the Myo accelerometer (gravity roll around the forearm axis, with the
  gain calibrated per subject on the labelled pro/supination movements). In DB1, imputed from movement labels.
- **wrist flexion in DB1**: imputed from labels, because the wrist-pitch sensor duplicates sensor 12.

The bank currently holds **15.3 h** of real intent: DB1 5 subjects (6.5 h) and DB5 10 subjects (8.8 h).
Copy more DB1 subjects into `data/real/ninapro_db1/` and rerun `myogen ninapro-bank` to grow it.

### Devices (`generate.device: raw | otto_bock | myo`)

| | layout | front-end | output |
|---|---|---|---|
| `otto_bock` (DB1) | 8-electrode ring at the radio-humeral level + 2 over the FDS/EDC bellies | 90–450 Hz, rectify, 3–6 Hz smoothing | 10 ch, 100 Hz, 0–5 V |
| `myo` (DB5) | 2 rings of 8, second distal and rotated 22.5° | aliased or anti-aliased 200 Hz sampling, ADC noise | 16 ch, 200 Hz, int8 counts |
| `raw` | 1–2 rings, any count | 20–450 Hz amplifier | mV at `generate.fs_emg` |

### Arm support (`generate.support: random | supported | free`)

NinaPro subjects rest the forearm on a support. `supported` sessions carry the static load (gravity, passive
muscle forces and joint springs) on the elbow, part of pronation/supination and part of the wrist, so muscles only
produce movement. `free` is an arm held up, which is the prosthesis-use case with tonic elbow-flexor EMG.
`random` mixes them 50/50.

## 3. Realism against NinaPro (`src/myogen/realism.py`)

Both sides go through the same pipeline: 100 Hz envelopes, per-subject 99th-percentile normalisation, NinaPro rest
labels. Simulated data uses NinaPro motion from the same database, the emulated device, supported arms and no
amputees. Latest results:

| statistic | DB1 real (range) | DB1 sim | DB5 real (range) | DB5 sim |
|---|---|---|---|---|
| ring correlation at 1/2/3/4 electrodes | .65 .47 .45 .42 | **.61 .47 .43 .48** | .72 .60 .58 .57 | **.79 .63 .55 .53** |
| participation ratio | 2.63 (2.52–2.95) | **2.59 ✓** | 2.12 (1.56–2.56) | **2.04 ✓** |
| EMG→movement lag (ms) | 62 (20–100) | **83 ✓** | 130 (60–190) | **109 ✓** |
| ridge R² (envelope → latent) | 0.23 (0.20–0.26) | 0.28 ✗ (slightly easier) | 0.32 (0.26–0.39) | **0.27 ✓** |
| median frequency (Hz) | – | – | 66.4 (63–68) | **66.8 ✓** |
| clipping | – | – | 0.000 | **0.000 ✓** |
| envelope autocorr 50 ms | 0.961 (0.945–0.97) | **0.948 ✓** | 0.974 (0.97–0.98) | 0.954 ✗ |
| envelope autocorr 200 ms | 0.85 (0.81–0.88) | 0.70 ✗ | 0.87 (0.84–0.90) | 0.68 ✗ |
| rest / movement ratio | 0.24 (0.15–0.33) | 0.53 ✗ | 0.29 (0.22–0.35) | 0.55 ✗ |
| \|x\| ≤ 1 count at rest | – | – | 0.55 (0.46–0.62) | 0.79 ✗ |

**What passes is the physics and geometry:** spatial structure, dimensionality, timing, spectrum and
decodability. **What fails is behaviour:** real subjects make larger and more sustained efforts during NinaPro
movements relative to rest than the unloaded simulated hand does (lower rest/move contrast, faster envelope
fluctuations). The knobs are `hold_tone`, `cocontraction_*`, `load_prob` and the grip-load magnitude in
`subject.py` / `generate.py`.

### How the gates moved (each fix was a diagnosed cause, not a fitted constant)

| Fix | Cause found |
|---|---|
| metrics: interleaved CV, onset lag | the chronological split and raw xcorr made the *real* data look undecodable |
| armrest support | tonic BRA/BRD/PT/PQ holding the forearm against gravity |
| seated shoulder pose in the model | overstretched TRIlong → 7–9 Nm passive elbow torque |
| engagement-gated co-contraction | co-contraction episodes during rest (c ≈ 0.08 everywhere) |
| realistic noise / gain, Otto Bock 90–450 Hz front-end | noise + mains made up most of the "rest" envelope |
| leaky smoothness | redundant co-activation decayed over ~3 s after each movement |
| device-specific decay / noise, holding tone, Otto Bock smoothing | ring correlation, Myo rest floor, envelope delay |

## 4. Motor-unit-level EMG with BioMime (`generate.emg_model: biomime`)

The analytic source model (band-limited noise per muscle) is fast but has no motor units. The BioMime backend
(`src/myogen/biomime/`) reimplements the NeuroMotion idea (Ma et al. 2024, `docs/`) on our MuJoCo model:
**spike trains x dynamic MUAPs**, with every intermediate variable available.

```bash
pip install "myogen[biomime] @ ..." --extra-index-url https://download.pytorch.org/whl/cpu
myogen download muap-bank                       # or: download biomime-weights + biomime-bank build (~10 min)
myogen generate --config configs/val_db5_biomime.yml
myogen biomime-bank info                        # territories, unit counts, gains
```

| Stage | Module | What it does |
|---|---|---|
| anatomy | `biomime/anatomy.py` | maps our 48 muscles into BioMime's forearm: 8 anchor compartments use NeuroMotion's published territories; others use our line-of-action geometry rotated onto the anchors (~33 deg RMS), plus 2 anatomical overrides (BRD, FPL). Deep muscles beyond BioMime's 23 mm training depth are clipped and attenuated (6 mm e-folding, BioMime's own decay). Upper-arm and intrinsic hand muscles are excluded |
| MUAP bank | `biomime/bank.py` | 128 motor units per muscle (fewer if the muscle has fewer) spanning the size distribution (NeuroMotion Eq. 1-2), each rendered at 5 fibre-length bins (L = 0.85-1.15; depth ~ L^-1/2, CV ~ L^-1, length ~ L), full 10x32 grid, float16 uV. BioMime runs once (~28 ms/MUAP on CPU) |
| electrodes | `biomime/electrodes.py` | any layout as poles on the periodic grid: rings, staggered rings, targeted, HD grids; bipolar along the fibres with per-device inter-electrode distance; extra fat -> spatial low-pass + attenuation |
| motor-unit pools | `biomime/mupool.py` | size-ordered pools behind one interface: `FuglevandPool` (rate model, vectorised; ~9->36 Hz, ISI CV 0.14) and `LIFPool` (leaky integrate-and-fire, ~15->28 Hz, CV 0.16-0.3). New neuron models subclass `MotorUnitPool` |
| synthesis | `biomime/synth.py` | per subject: draws 96 of each muscle's bank units (subject-specific); per session: projects MUAPs onto the electrodes and compresses each channel's templates (rank 19-32 for 99.9 % energy); per trial: spikes are split between the two nearest fibre-length bins, then one sparse product + one batched FFT convolution. Fibres per muscle follow PCSA x NeuroMotion's cadaver-corrected density (40k-190k); units beyond BioMime's 870-fibre range scale linearly |
| acquisition | `emg.acquire` | the same noise / mains / artifact / amplifier / device chain as the analytic model |

With `generate.save_spikes: true`, each trial also stores `spikes/{unit, sample}` (2048 Hz) and `spikes/units/{muscle, rank,
threshold, size, cond, weight}`: ground truth for decomposition algorithms and for training or driving spiking
networks. `labels/fibre_len` (normalised fibre length per muscle) is stored for every model.

**Realism against DB5 (Myo emulation, same protocol as section 3).** At physical amplitude (`generate.biomime_gain: 1`)
the signal sits near the Myo's ADC floor and 2/9 gates pass. With `biomime_gain: 6`, 4/9 pass, and the
spatial/spectral statistics are the closest of either model:

| statistic | DB5 real | analytic | BioMime (gain 6) |
|---|---|---|---|
| ring correlation at 1/2/3/4 electrodes | .72 .60 .58 .57 | .79 .63 .55 .53 | **.73 .58 .54 .51** |
| median frequency (Hz) | 66.4 | 66.8 | 65.8 |
| EMG→movement lag (ms) | 130 | 109 | **130** |
| participation ratio | 2.12 | 2.04 | 2.23 |
| amplitude quantiles p5…p95 | .05 .07 .09 .21 .57 | .08 .10 .13 .21 .53 | .06 .08 .10 .16 .48 |
| envelope autocorr 200 ms | 0.87 | 0.68 | 0.59 ✗ |
| ridge R² | 0.32 | 0.27 | 0.15 ✗ |
| rest / movement ratio | 0.29 | 0.55 | 0.64 ✗ |

The remaining failures are the same behavioural gap as the analytic model (sustained effort during NinaPro
holds), amplified by motor-unit discreteness: a few large units near an electrode make the envelope burstier than
Gaussian interference. Gain 6 is a device/effort calibration, not physics: physical amplitudes are ~0.07 mV RMS at
30 % drive and ~0.26 mV at full drive. Figures: `reports/biomime/` (spike raster, MUAPs vs fibre length).

**Performance**: bank build ~7 min (once); per session 3 s (projection + low-rank basis); 30 s of EMG at 30 %
drive on all muscles in 3.4 s. Low-rank synthesis is within 5 % of exact shift-and-add (`tests/`).

**Assumptions to keep in mind**
- BioMime encodes **one** forearm anatomy and a fixed 10x32 grid. Its axial placement on our forearm is not
  documented, so `GridGeometry` assumes a 10 mm row pitch from 6 to 42 % of forearm length. Recalibrate it with
  `myogen biomime-bank regrid` (MUAPs unchanged).
- Between-subject variability comes from motor-unit sampling, pool parameters, skin, electrodes and devices. It
  does not come from anatomy: that would need a retrained or anatomy-conditioned BioMime.
- The public checkpoint has 6 conditions (no fat conductivity); skin is modelled outside the network.
- **License and citation:** BioMime's generator network is vendored in `src/myogen/biomime/vendor/`
  (GPL-3.0, unmodified apart from one import; licence text, source commit and BibTeX in its `README.md`). The
  pretrained weights are downloaded, not committed. The NeuroMotion territory table in `anatomy.py` is cited
  data; no NeuroMotion code is used. Because GPL-3.0 code is part of the repository, distributing it means
  distributing it under GPL-3.0-compatible terms (citing is not enough on its own).

## 5. Next steps

1. **Fit the behaviour knobs automatically**: a small Bayesian or random search over hold tone, co-contraction
   and grip-load ranges against the gates. They are cheap to evaluate (≈ 3 min per device).
2. **Myo firmware notch**: the real DB5 PSD has a 50 Hz notch the emulation lacks.
3. **The decisive test**: pretrain on synthetic, fine-tune on DB5, and compare against training from scratch,
   measuring R² against calibration seconds per held-out subject.
4. **BioMime backend**: fit `generate.biomime_gain` and the grid placement (`myogen biomime-bank regrid`) with the realism gates,
   together with the behaviour knobs of item 1; grow the bank (more units per muscle, several latent draws) for
   more subject diversity; use the saved spikes for decomposition ground truth and LIF / SNN decoders.

## 6. Data format

`<out>/shard_XXX.h5`, one group per trial (`sSSSS_xXX_tTT`): `emg` (device units at `attrs.fs_emg`) and
`labels/{latent, q, qd, q_ref, act, exc, reserve, cocontraction, grip_load, restimulus}` at 100 Hz
(`restimulus` = NinaPro movement label for real-motion trials, −1 for synthetic). Attributes hold the full
`subject_params`, `session_params`, `meta` (electrode positions, gravity, motion source) and `qc`.
`manifest.jsonl` has one line per trial. QC: tracking RMSE < 5°, wrist reserve < 0.02 Nm, saturation < 5 %.
