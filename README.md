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
  simulation/    musculoskeletal → muscle_activity → emg → acquisition
  tools/         calibration, viewer, command line
  params.py      the model's constants
  simulator.py   runs the pipeline
  recording.py   the output
```

## License

MIT. The forearm model in `src/myogen/models/` is derived from MyoSim and distributed under Apache-2.0
(`src/myogen/models/LICENSE-MyoSim`).
