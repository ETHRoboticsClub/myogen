# myogen — forearm sEMG from hand and wrist joint angles

**In:** hand and wrist joint angles over time. **Out:** the multichannel surface EMG that electrodes on the forearm
would record, plus what the muscles did. In between, a MuJoCo forearm model (24 joints, 48 muscles) is driven by
its muscles along your motion, and an EMG engine turns the muscle activity into EMG at electrodes you place.

```
joint angles ──► ForearmModel ──► MuscleController ──► EMG engine ──► Electrodes.record ──► Recording
 (HandMotion)    gravity, armrest,  excitations that     analytic or     noise, amplifier,     EMG + labels
                 passive springs    reproduce the motion  BioMime         device              (HDF5)
```

## Install

```bash
pip install -e .                    # the simulator (analytic engine)
pip install -e ".[viewer]"          # + the web viewer (myogen replay)
pip install -e ".[biomime]" --extra-index-url https://download.pytorch.org/whl/cpu   # + the BioMime engine
myogen biomime-setup                # once: download the BioMime weights, build the MUAP bank (~10 min)
```

## Use

```python
import myogen as mg

motion = mg.HandMotion.load("motion.h5")                         # or mg.JointAdapter(...)(angles, fs), below
electrodes = mg.Electrodes.ring(n=8, position_cm=7)              # or .myo(), .otto_bock(), your own list
recording = mg.Simulator(electrodes, engine="biomime").run(motion)
recording.save("emg.h5")
mg.replay(recording)                                             # http://localhost:8080
```

```bash
myogen example motion.h5                                         # an example motion (open-close grasps)
myogen electrodes myo my_electrodes.yml                          # a preset as an editable file
myogen simulate motion.h5 emg.h5 --engine biomime --electrodes my_electrodes.yml
myogen replay emg.h5
```

### 1. Input: joint angles (`motion.py`)

`HandMotion` holds the 23 hand/wrist joints of the model (`myogen.body.HAND_JOINTS`, radians) at any sample rate,
optionally the elbow angle and where the hand is at rest. Files are `.h5` / `.npz` with `angles (T, J)`,
`joint_names (J,)`, `fs` and optionally `rest (T,)`.

If your joints are named or signed differently, a `JointAdapter` maps them:

```python
adapter = mg.JointAdapter({"index_mcp": "mcp2_flexion_r",
                           "index_pip": "pm2_flexion_r",
                           "wrist_fe": ("flexion_r", -1.0, 0.0)},     # (model joint, sign, offset in rad)
                          degrees=True)
motion = adapter({"index_mcp": a, "index_pip": b, "wrist_fe": c}, fs=120)
```

Joints you do not measure are filled and listed in `motion.filled` (and in the recording): a finger's MCP and PIP
from each other, its DIP from its PIP (DIP = 0.725 PIP, the joints are coupled), everything else at a relaxed
posture. Angles beyond the model's joint limits are clipped with a warning.

### 2. Electrodes (`electrodes.py`)

Every electrode is bipolar and placed in anatomical terms, the same for both engines:

| field | meaning |
|---|---|
| `angle_deg` | around the forearm: 0 radial border, +90 dorsal (extensors), −90 palmar (flexors), ±180 ulnar |
| `position_cm` | distance from the elbow (the model's forearm is 25.8 cm) |
| `ied_mm` | distance between the two poles |
| `orientation_deg` | pole axis: 0 along the forearm (the fibres), 90 around it |

`Electrodes` adds the device: `raw` (mV at `fs`), `myo` (NinaPro DB5 armbands: int8 counts at 200 Hz) or
`otto_bock` (NinaPro DB1: rectified, smoothed envelope in V at 100 Hz), and the recording noise (`noise_uv`,
mains, movement artifacts). Presets: `ring`, `myo` (two 8-pod bands at 3.1 / 6.6 cm, the second rotated 22.5°),
`otto_bock` (8 around at 2.6 cm + 2 over the FDS / EDC bellies).

### 3. Simulator and engines (`simulator.py`, `engines/`)

`Simulator(electrodes, engine, subject)` prepares the engine once; `run(motion)` returns a `Recording`. The
`Subject` defaults to an average adult (forearm radius, fat, posture, armrest, effort); `Subject.random(seed)`
draws one from population ranges. Same motion, electrodes, subject and seed give the same recording.

| | `analytic` | `biomime` |
|---|---|---|
| EMG model | band-limited noise sources along each muscle belly, attenuated with distance to each pole; the second pole sees the activity delayed by its travel time along the fibres | motor-unit pools fire spikes; each spike adds a MUAP from the BioMime network (Ma et al. 2023, NeuroMotion 2024), changing with fibre length |
| Electrode placement | anywhere on the skin | anywhere around the forearm, **1.55–10.8 cm from the elbow** (the BioMime grid); outside it is an error |
| Extra output | — | motor-unit spike trains (`save_spikes=True`) |
| Needs | nothing | PyTorch + the MUAP bank (`myogen biomime-setup`) |
| Amplitude | calibrated (`Subject.gain_mv`) | physical at `gain=1`; ~6 matched NinaPro DB5 |

**Evidence and limits.** Neither engine is validated end to end. In realism checks against NinaPro DB5
(`git checkout toolbox-v0`, `reports/`), the analytic engine passed 5/9 statistics and BioMime 4/9; BioMime was
closer on spatial correlation and EMG-to-movement lag, worse on envelope smoothness. A paper backs BioMime's MUAP
network, not its use here: its grid placement along the forearm is an assumption (`GridGeometry`), it encodes one
forearm anatomy, muscles other than its 7 anchors are mapped from our geometry (33° RMS fit residual), and the
public weights have no fat condition (skin is modelled outside the network). Both engines place the anchored
muscles at the same angle around the forearm (`tests/test_biomime.py`), but they disagree by ~70° on FCU and FDS:
BioMime follows NeuroMotion's anatomy, the analytic engine MyoArm's lines of action.

### 4. Output and replay (`recording.py`, `viewer.py`)

`Recording.save` writes HDF5: `emg (N, C)` (attrs `fs`, `channel_names`); `labels/*` at 100 Hz (`q`, `q_ref`,
`qd`, `excitation`, `activation`, `fibre_length`, `reserve`, `effort`, `grip_load`; attrs `joint_names`,
`muscle_names`); `spikes/*` (BioMime); and the electrodes, subject, engine and QC as attributes. `qc["pass"]` says
whether the model followed the motion (tracking RMSE < 5°, no wrist reserves, no saturation).

`mg.replay(recording)` / `myogen replay emg.h5` loops the hand in the browser with the skin, the electrodes (by
channel or by EMG activity), the muscles coloured by excitation, gravity, the armrest and a scrolling EMG plot.

### 5. Matching a real device (`calibration.py`)

The electrode YAML is the **device profile**: placement, the device (`raw` / `myo` / `otto_bock`), an overall
`gain`, the noise, and the device's own processing (`frontend`: amplifier band, Myo ADC gain / noise / aliasing,
Otto Bock band / smoothing / output gain). Edit it by hand or fit it to recordings of that device:

```python
sim = mg.Simulator(mg.Electrodes.myo(), engine="biomime")
cal = mg.calibrate(sim, motions, real_emgs, rests)     # recordings of the device + what the hand did in each
print(cal)                                             # statistics: real / before / after
cal.electrodes.save("my_myo.yml")                      # use with --electrodes my_myo.yml
mg.compare(real_emg, recording.emg, fs, "myo", rest)   # the same statistics for any pair of recordings
```

`calibrate` fits only what a recording identifies about the device, each against one statistic: with
`fit_rotation=True` the armband rotation (the rotation whose per-channel envelopes correlate best with the real
ones; armbands are rarely donned at a recorded angle), then `gain` (95th-percentile envelope while moving),
`noise_uv` (envelope floor at rest; it also absorbs residual activity of subjects who do not fully relax) and, for
Otto Bock, `frontend.smooth_hz` (envelope autocorrelation at 50 ms). It does not fit behaviour (effort); its
statistics (rest/movement ratio, neighbour correlation, spectrum, per-channel correlation) are reported so you can
see what remains. Calibrate on some subjects and report
on others. The simulation runs once per recording; each candidate setting only re-runs the recording chain.

### NinaPro example (`scripts/ninapro_compare.py`)

Simulates a window of a NinaPro DB1 / DB5 file from its glove motion and compares it with the real EMG:

```bash
python scripts/ninapro_compare.py data/real/ninapro_db5/s1/S1_E2_A1.mat --engine biomime --start 30 --duration 20 \
    [--calibrate my_myo.yml [--fit-rotation]] [--electrodes profile.yml] [--subject-seed 3] --out results/db5_s1_e2
```

It writes the simulated recording (`myogen replay .../simulated.h5`), `comparison.json` and `comparison.png`
(real vs simulated EMG, envelopes, joint motion). The glove → joint-angle retargeting lives in the script, not in
the library: the CyberGlove is uncalibrated, so it is normalised per subject (all files in the folder), and
pronation comes from the Myo accelerometer (DB5) or the exercise-2 labels (DB1). Wrist and forearm movements are
captured less well than finger movements, which shows as real EMG during apparently still hands. The
accelerometer is the Myo armbands' own IMU (DB5 stores it as `acc`); the script fades pronation back to rest during
labelled rest, which also removes accelerometer transients such as the one at the start of S1_E1.

## The model

`src/myogen/models/myoforearm_r.xml` is built from MyoSim's MyoArm (`scripts/build_model.py`, needs
`myo-sim`): MyoHand's 39 muscles plus the proximal forearm and elbow muscles a NinaPro-style armband sits over
(SUP, BRD, ANC, BIC, BRA, TRI), with the shoulder welded arm-by-the-side (MyoSim's preview pose stretches TRIlong
into a permanent 7–9 Nm passive elbow torque). The thumb lacks its intrinsic muscles (invisible to forearm
electrodes anyway); cheap reserve torques stand in for them.

The controller (`muscles.py`) is computed muscle control at 100 Hz: inverse dynamics, then bounded least squares
over the 48 activations (effort, co-contraction, smoothness, reserves), then inverse activation dynamics. Passive
joint springs make the relaxed posture the passive equilibrium, so a relaxed hand is nearly silent; an armrest
carries the static torques. That is the mechanical minimum, and a light unloaded finger needs almost none, while
people drive their muscles much harder. This **effort** (`Subject.hold_tone`, `move_cocontraction`,
`cocontraction`) is added to the neural drive as torque-neutral co-activation of the muscles of the joints that
move or are held away from rest, weighted by their moment arms: it reaches the EMG without changing the motion
(balanced co-contraction stiffens a joint without moving it). Grip load only appears while the hand is closed.

## Layout

| file | block |
|---|---|
| `motion.py` | `HandMotion` (input), `JointAdapter` (your joint names → the model's) |
| `electrodes.py` | `Electrode`, `Electrodes`, `Frontend` (placement, presets, recording hardware = the device profile) |
| `simulator.py` | `Simulator`: the pipeline (`run` = `record(simulate_clean(motion))`) |
| `calibration.py` | `calibrate`, `compare`: device profiles fitted to real recordings |
| `subject.py` | `Subject`: the simulated person |
| `body.py` | `ForearmModel`: MuJoCo model, forearm frame, gravity / armrest / passive springs |
| `muscles.py` | `MuscleController` → `MuscleActivity` |
| `engines/analytic.py` | `AnalyticEngine` |
| `engines/biomime/` | `BioMimeEngine` (`engine.py`), network, anatomy mapping, MUAP bank, motor-unit pools; `vendor/` is BioMime's network (GPL-3.0) |
| `recording.py` | `Recording`: output, HDF5 |
| `viewer.py` | `replay()` |
| `cli.py` | `myogen` command |

**License note:** BioMime's network is vendored unmodified (apart from one import) in
`src/myogen/engines/biomime/vendor/` under GPL-3.0 (licence, source commit and BibTeX in its `README.md`);
distributing this repository means distributing it under GPL-3.0-compatible terms. The weights are downloaded
from the authors, not committed. The NeuroMotion territory table in `anatomy.py` is cited data.

The dataset generator, NinaPro tooling and realism checks of the earlier toolbox are in the `toolbox-v0` tag.
