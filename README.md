# myogen

Synthetic forearm surface EMG from hand and wrist joint angles, on a MuJoCo musculoskeletal forearm model.

## Install

```bash
pip install -e .                    # the library
pip install -e ".[viewer]"          # + the 3-D web viewer (myogen replay)
pip install -e ".[examples]"        # + what the notebooks need
pip install -e ".[dev]"             # + tests, linting, type checking
```

## Use

```python
import myogen as mg

motion = mg.example_motion(duration=4.0)              # or mg.HandMotion.load("motion.h5", mg.read_motion_file)
recording = mg.Simulator(mg.Electrodes.myo()).run(motion)
recording.save("emg.h5")
```

The notebooks are the way in: `notebooks/01_quickstart.ipynb`, then `notebooks/02_customising.ipynb`.

## Layout

```
src/myogen/
  inputs/        motion, subject, electrodes
  simulation/    one block per step, each with its input and output:
                   kinematics      HandMotion → JointTrajectory
                   muscle_solver   JointTrajectory → MuscleActivity (least squares on musculoskeletal)
                   effort          + co-activation, grip and rest tone (heuristics)
                   emg             MuscleActivity → clean EMG, through:
                     motor_units     drive → Spikes; where each unit sits (UnitPlacement)
                     geometry        the subject's forearm, poles, atlas anatomy (anatomy)
                     muap            what each pole sees of a unit (biphasic | layered volume conductor)
                     montage         pole signals → channels
                   acquisition     clean EMG → the device's output
  tools/         calibration, viewer, command line
  params.py      the model's constants
  simulator.py   runs the pipeline
  recording.py   the output
```

## License

MIT. The forearm model in `src/myogen/models/` is derived from MyoSim and distributed under Apache-2.0
(`src/myogen/models/LICENSE-MyoSim`). The muscle anatomy (`src/myogen/models/forearm_anatomy.npz`, `docs/anatomy/`)
is derived from BodyParts3D, (c) The Database Center for Life Science, under CC BY-SA 2.1 Japan
(`src/myogen/models/LICENSE-BodyParts3D`; how it is made: `docs/anatomy.md`).
