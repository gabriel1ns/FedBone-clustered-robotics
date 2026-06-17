# RoboMimic integration

RoboMimic is the main offline robotic dataset path for this project.

The FedBone runner treats each RoboMimic HDF5 file as one manipulation task and
trains an imitation-learning objective:

```text
observation -> demonstrated action
```

## Expected Files

Place low-dimensional RoboMimic datasets under `data/robomimic/`.

Example:

```text
data/robomimic/
  lift/ph/low_dim.hdf5
  can/ph/low_dim.hdf5
  square/ph/low_dim.hdf5
  transport/ph/low_dim.hdf5
```

## Downloading Datasets

Install the Hugging Face CLI:

```bash
python -m pip install huggingface_hub
```

Windows `cmd.exe` one-line command:

```bat
hf download robomimic/robomimic_datasets v1.5/lift/ph/low_dim_v15.hdf5 v1.5/can/ph/low_dim_v15.hdf5 v1.5/square/ph/low_dim_v15.hdf5 --repo-type dataset --local-dir data/robomimic
```

PowerShell multi-line command:

```powershell
hf download robomimic/robomimic_datasets `
  v1.5/lift/ph/low_dim_v15.hdf5 `
  v1.5/can/ph/low_dim_v15.hdf5 `
  v1.5/square/ph/low_dim_v15.hdf5 `
  --repo-type dataset `
  --local-dir data/robomimic
```

The loader expects the standard RoboMimic HDF5 layout:

```text
/data/demo_0/obs/<obs_key>
/data/demo_0/actions
/data/demo_1/obs/<obs_key>
/data/demo_1/actions
```

By default, image-like observation keys are ignored and low-dimensional numeric
observation keys are flattened and concatenated.

## Configuration

In `config/config.py`:

```python
DATASET = "robomimic"
ROBOMIMIC_TASK_FILES = []
ROBOMIMIC_OBS_KEYS = []
ROBOMIMIC_TEST_RATIO = 0.2
ROBOMIMIC_SUCCESS_THRESHOLD = 0.05
```

Leave `ROBOMIMIC_TASK_FILES` empty to discover all `.hdf5` / `.h5` files under
`data/robomimic`. Set it explicitly to control task order or use only a subset.

Set `ROBOMIMIC_OBS_KEYS` when auto-detection picks the wrong low-dimensional
keys.

## FedBone Mapping

- Task: one RoboMimic HDF5 file, such as Lift or Can.
- Client: one robot-task shard.
- Input: flattened low-dimensional observations.
- Target: action vector from demonstrations.
- Loss: MSE.
- TSR: percentage of predicted actions within the configured action-error
  threshold.

## Main Comparison

The FedBone runner compares:

- FedBone with simple server-gradient averaging.
- FedBone with GP Aggregation.

FedAvg and Clustered FL should be adapted to this same action-regression
protocol before final external baseline comparison.
