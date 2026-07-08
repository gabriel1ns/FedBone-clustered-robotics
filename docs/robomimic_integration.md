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
  tool_hang/ph/low_dim.hdf5
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

PowerShell command for the current expanded `ph` scope:

```powershell
.\scripts\download_robomimic_lowdim.ps1 -Scope expanded-ph
```

Available script scopes:

- `core-ph`: Lift, Can, and Square with proficient-human demonstrations.
- `expanded-ph`: `core-ph` plus Tool Hang and Transport.
- `expanded-variants`: `expanded-ph` plus available `mh` / `mg`
  variants for broader task and demonstration heterogeneity.

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
ROBOMIMIC_TASK_FILES = []
ROBOMIMIC_OBS_KEYS = []
ROBOMIMIC_TEST_RATIO = 0.2
ROBOMIMIC_SUCCESS_THRESHOLD = 0.05
```

Leave `ROBOMIMIC_TASK_FILES` empty to discover all local `.hdf5` / `.h5` files
under `data/robomimic`. The loader ignores Hugging Face `.cache` files to avoid
training on duplicate downloads. Set `ROBOMIMIC_TASK_FILES` explicitly to
control task order or use only a subset.

Set `ROBOMIMIC_OBS_KEYS` when auto-detection picks the wrong low-dimensional
keys.

## FedBone Mapping

- Task: one RoboMimic HDF5 file, such as Lift or Can.
- Expanded task names include the demonstration type, such as `can_ph` or
  `can_mh`, so per-task metrics remain distinct when variants are used.
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

FedAvg and Clustered FL use this same normalized action-regression protocol.
Because tasks have different observation and action dimensions, each baseline
trains one federated model per task and reports both per-task and macro metrics.
FedBone is the approach that additionally shares its server backbone across
tasks.

## Online evaluation

Evaluate every RoboMimic task stored in a FedBone checkpoint:

```powershell
python runner/evaluate_online.py --task all --episodes 10
```

Use a single task or a comma-separated subset when needed:

```powershell
python runner/evaluate_online.py --task lift_ph,can_ph --episodes 10
```

The evaluator recognizes `ph`, `mh`, and `mg` variants of Lift, Can, Square,
Tool Hang, and Transport. When multiple tasks are evaluated, the output JSON
contains per-task results and macro success-rate / return summaries. If
`--record-video` is set, the task name is appended to each video filename.
