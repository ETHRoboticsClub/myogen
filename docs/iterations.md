# Iterations: myogen channel by channel against free-movement recordings

Goal: with the default subject, myogen replaying recorded hand kinematics (free movement, kin -> EMG; forearm rotation
only when tracked) gives, on every electrode, the EMG the recording has.

## Method

- **Target**: the free-movement sessions of EMG-FK, NinaPro DB2 / DB7 (exercise B), DB8, intact subjects with a known
  band placement. Subjects drawn at random per dataset (seed 0): the first 3 are the **tune** set (every decision),
  the next 3 the **validate** set (reported, never tuned on). Dataset repo, `src/femg/config/synthetic.yaml`
  (`target`).
- **Score** (`compare_synthetic.score`), per dataset, median over sessions:
  - *channel*: r between real and myogen of each ring channel's RMS over time (log envelope), median over channels;
  - *profile*: r of the channels' levels while moving (log RMS per channel over the ring mean);
  - *score* = their mean; the table's mean is over datasets. Also reported: ring-pattern r per movement and the R² of
    a joint decoder trained on myogen and tested on the recording.
- **Electrode position**: each real band is turned into the calibration baseline's frame (`sessions.parquet`: its
  rotation and direction found from the recordings alone). myogen's ring electrodes are placed in that frame at
  `ring_offset_deg` (a continuous turn of the simulated electrodes, not a channel shuffle) and `ring_reversed` (the
  frame's direction), fitted once on the tune set. Electrodes placed on a muscle by palpation (DB2 / DB7 FDS, EDC) keep
  their anatomical angle.
- **Speed**: the muscle activity of every session is cached apart from the EMG; an EMG variant re-runs only the EMG
  step (seconds per session). `scripts/tests/iterate_synthetic.py variants.yaml` scores a list of variants.

## Log

### 1. Electrode frame (ring_offset_deg)

Baseline (atlas, motor units, biphasic waveforms, decay 8 mm, default subject): channel r 0.24, profile r 0.01 (tune).
The channels follow their recordings over time, but their levels around the ring do not match at all: a frame
problem. Sweep of the offset in 45° steps, both directions, then 11° steps around the best:

| tune | score | EMG-FK | DB2 | DB7 | DB8 | channel | profile | pattern | decoder |
|---|---|---|---|---|---|---|---|---|---|
| offset 0° | 0.126 | 0.152 | 0.172 | 0.179 | -0.001 | 0.240 | 0.011 | 0.145 | -0.133 |
| **offset -45°** | **0.483** | 0.501 | 0.566 | 0.444 | 0.419 | 0.237 | **0.728** | 0.178 | -0.306 |
| offset -45°, reversed | 0.368 | 0.426 | 0.401 | 0.266 | 0.378 | 0.220 | 0.516 | 0.015 | -0.419 |
| offset -56° / -34° | 0.464 / 0.442 | | | | | | 0.704 / 0.660 | | |

Validate: score 0.266 -> **0.467**, profile 0.316 -> 0.726, pattern 0.142 -> 0.297, decoder -0.185 -> -0.113, every
dataset better. Adopted: `ring_offset_deg: -45`. The calibration baseline's palmar (where wrist flexion peaks) then
sits at myogen -135°, over the atlas's flexor mass (ulnar-palmar in a supinated forearm): one rule, found from the
recordings, and one turn, the same for every dataset.

### 2. EMG spatial model and motor units (EMG only)

From iteration 1 (offset -45°), tune:

| variant | score | channel | profile | pattern | decoder |
|---|---|---|---|---|---|
| baseline: biphasic, decay 8 mm | 0.483 | 0.237 | 0.728 | 0.178 | -0.306 |
| decay 6 / 10 / 12 / 16 mm | 0.451 / 0.502 / 0.511 / 0.518 | 0.215 / 0.257 / 0.267 / 0.280 | ~0.69-0.76 | | |
| volume conductor (electrode 2.5 or 1 mm) | 0.471 | 0.273 | 0.669 | 0.171 | -0.353 |
| volume conductor, territory 4 mm | 0.487 | 0.285 | 0.690 | 0.200 | -0.355 |
| recruitment over 0.9 of the drive (0.7) / 0.5 | 0.502 / 0.468 | | | | |
| unit size range 30 (10) | 0.498 | | | | |
| peak rate 25 Hz, rate gain 20 | 0.479 | | | | |
| decay 16 + recruitment 0.9 | 0.529 | 0.298 | 0.761 | 0.205 | -0.414 |
| ... + size range 30 | 0.531 | 0.311 | 0.752 | 0.178 | -0.646 |
| volume conductor, territory 6 mm + recruitment 0.9 + size 30 | 0.515 | 0.322 | 0.709 | 0.196 | -0.808 |

Validate, decay 16 + recruitment 0.9 against the baseline: 0.470 vs 0.467 (EMG-FK 0.30 vs 0.40, pattern 0.21 vs
0.30, decoder -0.35 vs -0.11). **Not adopted**: the gain on the tune subjects does not hold, and wider spread costs the
pattern match and transfer. The EMG step's spatial and motor-unit settings are not what limits the per-channel
match: the channel r over time stays near 0.2-0.3 whatever they are. The drive (which muscles are active when) is
next.

### 3. Drive: rest tone, co-activation, smoothing

The simulated envelope's rest level is too close to its moving level (rest / move RMS 0.70-0.86 against 0.27-0.59
in the recordings). From iteration 1, tune (muscle activity re-simulated per variant):

| variant | score | channel | profile | pattern | decoder |
|---|---|---|---|---|---|
| baseline (rest tone 0.04, hold tone 0.15) | 0.483 | 0.237 | 0.728 | 0.178 | -0.306 |
| rest tone 0.08 | 0.429 | 0.194 | 0.664 | 0.116 | -0.245 |
| **rest tone 0.02** | **0.541** | 0.340 | 0.742 | 0.262 | -0.783 |
| rest tone 0.01 / 0.0 | 0.550 / 0.548 | 0.353 / 0.347 | 0.746 / 0.749 | 0.306 / 0.262 | -0.894 / -1.179 |
| rest tone 0.02, hold tone 0.25 | 0.539 | 0.340 | 0.739 | 0.243 | -0.653 |
| hold tone 0.05 / 0.25 | 0.469 / 0.494 | 0.206 / 0.261 | | | |
| move co-contraction 0 / 0.25, no co-contraction episodes | 0.479-0.486 | | | | |
| drive low-pass 2 / 8 Hz | 0.486 / 0.483 | | | | |

Validate: rest tone 0.02 scores 0.528 against 0.467 (channel 0.300 vs 0.208, profile 0.756 vs 0.726, DB2 / DB7 / DB8
all better, EMG-FK equal), rest tone 0.01 0.529. **Adopted: rest tone 0.02** (random subjects 0.01-0.04). Only the
rest level matters; co-activation and drive smoothing do not move the score. The cost is the decoder trained on
myogen (-0.11 -> -0.41 on validate, lower still with less tone): with a quieter rest, the decoder learns that rest
is silent, and the recordings' rest is not. Kept at 0.02 rather than 0.01 for that reason.

### 4. Motor-unit count and common drive (EMG only)

The simulated envelope is jumpier than the recorded one (autocorrelation at 200 ms 0.42-0.56 against 0.75-0.82): more
units should average it out. From iteration 3, tune:

| variant | score | channel | profile | pattern | decoder |
|---|---|---|---|---|---|
| baseline (0.25 units / N, 20-200) | 0.541 | 0.340 | 0.742 | 0.262 | -0.783 |
| units x2 / x4 / x8 | 0.533 / 0.540 / 0.545 | 0.358 / 0.354 / 0.356 | 0.708 / 0.727 / 0.734 | | |
| no common drive | 0.540 | 0.338 | 0.741 | 0.279 | -0.827 |
| units x4, no common drive | 0.540 | 0.353 | 0.727 | 0.283 | -0.700 |

**Not adopted**: within ±0.01, and 4-8 times the EMG cost. The envelope's jumps are in the drive, not in the
unit sampling.

### 5. Controller and effort (muscle activity re-simulated)

From iteration 3, tune: effort weight 1e-3 / 1e-2 (default 3e-3) 0.542 / 0.544, smooth leak 0.97 0.538, PD bandwidth
4 Hz 0.537, speed engaged 0.25 0.539, posture engaged 0.2 with dead zone 0.05 0.521; baseline 0.541. **Not adopted**:
the score does not depend on how the controller shares the work.

### 6. What limits the channel r

Two references added to the score, the channel r against: myogen's ring mean put on every channel (*common*) and the
hand's speed (log, 250 ms mean; *speed*), no EMG model at all. Tune, from iteration 3:

| variant | score | channel | common | speed | profile | pattern | decoder |
|---|---|---|---|---|---|---|---|
| baseline (decay 8) | 0.541 | 0.340 | 0.360 | 0.295 | 0.742 | 0.262 | -0.783 |
| decay 12 / 16 / 24 | 0.548 / 0.548 / 0.535 | 0.342 / 0.341 / 0.331 | 0.360 | | 0.753 / 0.756 / 0.738 | 0.190 / 0.183 / 0.179 | -1.0 / -1.2 / -1.3 |
| decay 16, recruitment 0.9 | 0.555 | 0.354 | 0.373 | | 0.756 | 0.230 | -1.772 |
| volume conductor (territory 2 / 6 mm) | 0.529 / 0.533 | 0.346 | 0.365 | | 0.713 / 0.720 | 0.142 / 0.153 | -1.2 / -1.4 |
| volume conductor, fat 0.1 S/m, skin 2 mm | 0.524 | 0.345 | 0.364 | | 0.703 | 0.123 | -1.347 |

Whatever the setting, myogen's channels match a recording's channels over time no better than their own ring mean
does, and only a little better than the hand's speed: what a channel adds over its neighbours in time is not yet
right, while the level of each channel (profile) is. The ring-mean envelope is not late (best lag 0 to -50 ms) nor
less smooth (lag-1 autocorrelation 0.98-1.00 as recorded); it follows the recording's (r 0.5-0.6 on NinaPro, 0.3 on
EMG-FK), with a wider log range on DB2 / DB8 (log SD 1.0 vs 0.4-0.6). **Not adopted**: wider spread gains < 0.015
and costs the pattern r. The channel r is bounded by which muscles the kinematics alone imply at each moment (the
controller, iteration 5, does not move it): the next lever is the drive's muscle selection itself (e.g. the
muscle-to-joint moment arms of the hand model, the clipped finger extension), not the EMG step.

### 7. Training check (adopted settings: offset -45°, rest tone 0.02)

Cross-subject joint decoding (`train_synthetic.py`, tune + validate subjects, 6 per dataset, 3 random people per
session), R² on the held-out subject, median:

| | real | real + myogen | myogen | myogen, own movements |
|---|---|---|---|---|
| EMG-FK | -0.11 | -0.17 | -0.29 | -0.28 |
| DB2 | -0.13 | **-0.09** | -0.15 | 0.06 |
| DB7 | 0.20 | -0.00 | -0.11 | -0.11 |
| DB8 | 0.05 | -0.06 | -0.20 | -0.05 |

myogen data helps on DB2 only; elsewhere it still costs the decoder, as the decoder transfer in the score said. The
channel levels now match (profile r 0.75), their course over time does not (iteration 6), and a decoder reads the
course. Until the drive's muscle selection follows the recordings, myogen data is for pretraining and checks, not for
mixing into a cross-subject decoder's training set.
