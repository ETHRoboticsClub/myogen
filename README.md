# myoforearm — musculoskeletal forearm simulation and synthetic sEMG generator

A MuJoCo forearm/hand musculoskeletal model (built from MyoSim's **MyoArm**: 24 DoF, 48 muscles), a muscle-driven
controller that reproduces natural hand movement (synthetic or **real NinaPro glove trajectories**), and a sEMG
forward model that emulates the **NinaPro DB1 (Otto Bock)** and **DB5 (Myo)** recording hardware. The output is
physically consistent (intent, kinematics, muscle activity, multichannel EMG) data for pretraining the EMG → hand
model, plus a **realism report** that checks the synthetic EMG against the real datasets.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.build_model                  # models/myoforearm_r.xml (already committed)
.venv/bin/python -m src.ninapro bank                 # data/real/latent_bank.h5 from the NinaPro glove data

# pretraining data: mixed synthetic/real motion, mixed armrest/free arm, 30 % amputees
.venv/bin/python -m src.generate --out data/run1 --subjects 50 --sessions 3 --trials 4 --duration 30
# NinaPro-matched data + realism report
.venv/bin/python -m src.generate --out data/val_db5 --device myo --motion ninapro --bank-datasets db5 \
    --support supported --amputee-frac 0 --subjects 10 --sessions 2 --trials 3 --duration 40
.venv/bin/python -m src.realism --real db5 --sim data/val_db5 --out reports/realism_db5

.venv/bin/python -m src.sim demo | live | replay <shard.h5> [trial]   # viewer at http://localhost:8080
.venv/bin/python -m src.inspect_trial <shard.h5> [trial] --out trial.png
```

Throughput on this machine (16 cores): ~1.9× real time per worker, **~13 h of data per wall-clock hour** with 14
workers.

## Layout

| Path | Content |
|---|---|
| `src/build_model.py` | builds `models/myoforearm_r.xml` from MyoArm (pruning, seated shoulder pose) |
| `src/model.py` | model loading (`myoforearm` default, `myohand` legacy), joint/muscle bookkeeping by name, belly geometry |
| `src/motion.py` | 10-D posture latent, prototypes, `JointMap`, synthetic trajectory generator |
| `src/ninapro.py` | DB1/DB5 loaders, CyberGlove → latent retargeting, `LatentBank` of real intent trajectories |
| `src/tracker.py` | computed muscle control with reserves, co-contraction, leaky smoothness, armrest support |
| `src/emg.py` | volume-conductor geometry, source synthesis, device emulation (raw / Otto Bock / Myo) |
| `src/subject.py` | every randomised subject / session / device parameter |
| `src/generate.py` | parallel generation, passive equilibrium, QC, HDF5 shards |
| `src/realism.py` | realism gates against NinaPro |
| `models/` | the pinned model snapshot + meshes (editable, versioned with the code) |
| `data/real/ninapro_db1/` | 5 DB1 subjects copied from emg2action (raw `.mat`) |
| `data/real/ninapro_db5/` | all 10 DB5 subjects (`ninapro.hevs.ch/files/DB5_Preproc`; cite Pizzolato et al. 2017) |
| `reports/realism_db{1,5}/` | latest realism results (`realism.json`, `realism.png`) |

---

## 1. Model: why MyoArm-derived and not MyoHand

| | MyoHand (MyoSim) | **myoforearm** (this repo) |
|---|---|---|
| DoF | 23 (wrist + hand) | 24 (+ elbow, posture variable) |
| Muscles | 39 | 48: + **SUP, BRD, ANC** (bellies in the proximal forearm, i.e. right under the DB1/DB5 rings), BIC×2, BRA, TRI×3 |
| Supination | no supinator/biceps → reserve torques | muscle-driven (reserve 0.09 → ≤ 0.015 Nm) |
| Shoulder pose | 90° elevation preview pose | **seated, arm by the side**: TRIlong passive force 200–530 N → ~0; passive equilibrium elbow 65°, forearm −19° (physiological) |

`src/build_model.py` documents the pruning. The shoulder pose was chosen by a grid search minimising passive
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

### Motion sources (`--motion synthetic | ninapro | mix`)

`src/ninapro.py` retargets the 22-sensor CyberGlove to the same 10-D latent the generator uses. The sensor order
was verified against the single-finger movements of exercise 1. Two channels the glove does not measure are
recovered and flagged:

- **pronation/supination**: in DB5, from the Myo accelerometer (gravity roll around the forearm axis, with the
  gain calibrated per subject on the labelled pro/supination movements). In DB1, imputed from movement labels.
- **wrist flexion in DB1**: imputed from labels, because the wrist-pitch sensor duplicates sensor 12.

The bank currently holds **15.3 h** of real intent: DB1 5 subjects (6.5 h) and DB5 10 subjects (8.8 h).
Copy more DB1 subjects into `data/real/ninapro_db1/` and rerun `python -m src.ninapro bank` to grow it.

### Devices (`--device raw | otto_bock | myo`)

| | layout | front-end | output |
|---|---|---|---|
| `otto_bock` (DB1) | 8-electrode ring at the radio-humeral level + 2 over the FDS/EDC bellies | 90–450 Hz, rectify, 3–6 Hz smoothing | 10 ch, 100 Hz, 0–5 V |
| `myo` (DB5) | 2 rings of 8, second distal and rotated 22.5° | aliased or anti-aliased 200 Hz sampling, ADC noise | 16 ch, 200 Hz, int8 counts |
| `raw` | 1–2 rings, any count | 20–450 Hz amplifier | mV at `--fs-emg` |

### Arm support (`--support random | supported | free`)

NinaPro subjects rest the forearm on a support. `supported` sessions carry the static load (gravity, passive
muscle forces and joint springs) on the elbow, part of pronation/supination and part of the wrist, so muscles only
produce movement. `free` is an arm held up, which is the prosthesis-use case with tonic elbow-flexor EMG.
`random` mixes them 50/50.

## 3. Realism against NinaPro (`src/realism.py`)

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

## 4. Next steps

1. **Fit the behaviour knobs automatically**: a small Bayesian or random search over hold tone, co-contraction
   and grip-load ranges against the gates. They are cheap to evaluate (≈ 3 min per device).
2. **Myo firmware notch**: the real DB5 PSD has a 50 Hz notch the emulation lacks.
3. **The decisive test**: pretrain on synthetic, fine-tune on DB5, and compare against training from scratch,
   measuring R² against calibration seconds per held-out subject.
4. **Optional detailed EMG layer**: motor-unit-level synthesis (e.g. BioMime) behind the same
   `synthesize(u, G, …)` interface, if raw or HD-EMG realism becomes necessary.

## 5. Data format

`<out>/shard_XXX.h5`, one group per trial (`sSSSS_xXX_tTT`): `emg` (device units at `attrs.fs_emg`) and
`labels/{latent, q, qd, q_ref, act, exc, reserve, cocontraction, grip_load, restimulus}` at 100 Hz
(`restimulus` = NinaPro movement label for real-motion trials, −1 for synthetic). Attributes hold the full
`subject_params`, `session_params`, `meta` (electrode positions, gravity, motion source) and `qc`.
`manifest.jsonl` has one line per trial. QC: tracking RMSE < 5°, wrist reserve < 0.02 Nm, saturation < 5 %.
