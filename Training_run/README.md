# Train a MuJoCo Humanoid with PPO

`Training_run` trains Gymnasium's stock 17-motor Humanoid to move forward using
Proximal Policy Optimization (PPO). It includes configurable rewards, episode
logging, policy checkpoints, and video export.

**Edit [config.py](config.py) to change settings. Run all commands below from
the repository root—the folder containing `Training_run/`.**

## 1. Set up Python

If the project environment already exists, activate it:

```bash
source .venv/bin/activate
```

For a fresh environment, use a Python installation compatible with the pinned
packages (this project was tested with Python 3.10):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On an Apple Silicon Mac, use native ARM Python. Verify the active interpreter:

```bash
python -c "import platform, mujoco, stable_baselines3; print(platform.machine()); print('Imports work')"
```

It should print `arm64` on Apple Silicon. An environment created with Intel
Python needs to be recreated with native Python; installing packages again
inside it does not change its architecture.

## 2. Start training

Train for a specified number of episodes:

```bash
python -m Training_run train --episodes 10000 --run Training_run/runs/attempt_01
```

Or use `TRAIN_EPISODES` from `config.py`:

```bash
python -m Training_run train --run Training_run/runs/attempt_02
```

Omit `--run` to create a uniquely timestamped folder under `Training_run/runs/`.
An explicitly named output folder must not already exist.

`--episodes` counts **completed training episodes**, not action steps or
optimization updates. Evaluation trials do not count toward this budget.
Training starts from a randomly initialized policy each time; successful
running is not guaranteed by completing a particular episode count.

Progress prints approximately every 15 seconds, including completed episodes,
training steps, and an estimated remaining time. That ETA can change as episodes
get longer. It estimates the requested training budget, not time until the
humanoid learns to run.

### Stop training

Press **Ctrl-C** to stop and save the latest policy and normalization statistics.
The current CLI has **no resume command**. Interrupted checkpoints can still be
used for evaluation, but starting `train` again starts a fresh policy in another
folder. The video command uses the saved `best` checkpoint, not `latest`.

### PPO rollout detail

PPO updates after collecting full rollouts (`PPO_N_STEPS`, currently 512).
Stopping at exactly the requested episode count can leave a partial rollout;
Stable-Baselines3 discards that partial rollout without an optimization update.
Very short runs may finish before any update, and the script prints a warning.
Use more episodes for learning; shorter rollouts are useful for quick code tests.

## 3. Generate and watch a video

After training, export a video from its best saved policy:

```bash
python -m Training_run video --run Training_run/runs/attempt_01
```

The default output is:

```text
Training_run/runs/attempt_01/best_run.mp4
```

On macOS, open it with:

```bash
open Training_run/runs/attempt_01/best_run.mp4
```

To evaluate more candidates and choose a destination:

```bash
python -m Training_run video \
  --run Training_run/runs/attempt_01 \
  --episodes 10 \
  --seed 20000 \
  --output running.mp4
```

For automatic video export after training:

```bash
python -m Training_run train \
  --episodes 10000 \
  --run Training_run/runs/attempt_03 \
  --video
```

Video generation evaluates the best policy without exploration noise, selects
one candidate, then replays its seed to render it. A successful candidate is
preferred; among successful candidates, fewer steps is better. If all fail,
the candidate with greatest forward distance is selected. A JSON file beside
the video records the candidate results and selected seed.

**A best-case video is not a success-rate measurement.** Check the multi-episode
metrics too. Training does not render, but video export requires working graphics
access on the machine running it.

## 4. Adjust settings in one file

[config.py](config.py) contains plain Python variables. For example:

```python
TRAIN_EPISODES = 10000
ENV_FINISH_DISTANCE = 20.0
ENV_MAX_EPISODE_STEPS = 1000

REWARD_UPRIGHT_WEIGHT = 5.0
REWARD_FORWARD_VELOCITY_WEIGHT = 1.25
REWARD_FALL_PENALTY = 200.0
REWARD_FINISH_BONUS = 500.0
REWARD_FORWARD_REACH_WEIGHT = 0.5
REWARD_FORWARD_REACH_METERS = 0.35

PPO_LEARNING_RATE = 3.56987e-5
PPO_GAMMA = 0.95
```

| Prefix | Controls |
| --- | --- |
| `ENV_` | Finish distance, time limit, reset noise, simulation/control settings |
| `REWARD_` | Reward weights, posture/speed targets, knee and foot-reach shaping |
| `PPO_` | Learning rate, rollout and batch sizes, discounting, network architecture |
| `TRAIN_` | Episode count, seeds, evaluation frequency, CPU thread count |

Changes apply when a run starts. They do not alter an already-running process
or a saved policy. CLI `--episodes` overrides `TRAIN_EPISODES`; video options
can override its saved evaluation count and video seed.

For separate experiment settings:

```bash
cp Training_run/config.py Training_run/config_experiment.py
# Edit config_experiment.py, then:
python -m Training_run train \
  --config Training_run/config_experiment.py \
  --run Training_run/runs/experiment_01
```

Change one setting or a small related group at a time. Compare finish rate,
distance, survival, and videos. Do not compare raw returns as if they meant the
same thing after changing the reward equation. Episode budgets can produce
different training-step counts, so check `run.json` when comparing experiments.

## Episode rules

Each episode resets near the stock standing pose with small position and
velocity perturbations. The humanoid is unsupported and gravity is enabled.

| Ending | Condition | Default terminal reward |
| --- | --- | --- |
| Fall | Any non-foot body touches the floor | −200 |
| Finish | Center of mass advances 20 meters in world +X from the start | +500 |
| Timeout | 1,000 policy actions without a fall or finish | No extra terminal reward |

Contacts and finish distance are checked after each physics substep. Positive-
distance proximity contacts are not falls. A fall takes precedence over a
simultaneous finish. Sideways travel and loops do not accumulate finish distance.
There is no height-based or sideways-angle failure condition.

With the defaults, one action advances up to five 0.003-second physics steps:
about 66.7 actions per simulated second, or 15 seconds for a full episode.
A fall or finish can stop an action early. Falls and finishes are terminal
states; a timeout is reported as truncation.

## Reward components

The default `per_action` mode uses the following components. These are tunable
objectives, not guarantees of a natural gait.

| Component | Default behavior |
| --- | --- |
| Forward velocity | `1.25 ×` forward center-of-mass velocity |
| Standing height | Up to +5 near root height 1.4 m; Gaussian tolerance 0.15 m; zero on a fall |
| Heading | Forward alignment of torso/pelvis weighted 0.5 each, feet 0.25 each |
| Motor effort | Subtract `0.2 × sum(motor_commands²)` |
| External contact | Subtract `min(5e-7 × sum(external_force_and_torque_components²), 10)` |
| Forward foot reach | Small capped bonus for an advancing swing foot ahead of the pelvis |
| Fall / finish | One-off terminal penalty or bonus from the table above |

Core per-action terms are **not multiplied by elapsed time**. The standing-height
term measures root height, not head height or torso tilt.

Forward reach is a secondary objective, capped at 0.5 total per action. It
requires upright posture, forward motion, an unplanted foot with at least 3 cm
clearance, and forward foot motion relative to the pelvis. The reach target is
35 cm; exceeding it earns no extra reward. Both legs use the same formula and
their scores are averaged. Holding a foot out stops earning the bonus once it
stops advancing. This does not itself enforce alternating footfalls.

Additional distance, target-speed, sideways-motion, rhythmic-knee, and action-
change terms default to zero. Their settings are exposed for experiments.
Optional speed, sideways and knee terms are time-scaled, so their numerical
weights are not directly comparable with the core per-action weights.
`REWARD_HEADING_WEIGHT` only applies in `time_scaled` mode; in `per_action` mode,
use the individual torso, pelvis, and foot heading weights.

Enabling knee or action-change shaping automatically expands the observation
from 348 to 369 values so gait phase and previous actions are observable. Start
a fresh run when changing this; checkpoints with different input dimensions
are incompatible. The forward-reach bonus does not change observation size.

## Policy and learning settings

The default policy sees 348 simulator measurements including joint/root state,
body inertia and velocity, actuator forces, and external forces. It does not
use images. Its 17 outputs are motor commands in `[-0.4, 0.4]`; actuator gearing
converts them into torque. They are not desired joint angles.

PPO uses separate actor and critic networks with two 256-unit ReLU layers.
Defaults include 512-step rollouts, batches of 256, five optimization epochs,
learning rate `3.56987e-5`, discount `0.95`, and GAE lambda `0.9`. Configuration
also exposes exploration, gradient clipping and loss weights. `PPO_N_STEPS`
must be divisible by `PPO_BATCH_SIZE`, and both must be at least 2.

Observation and reward statistics are normalized during training. Evaluation
uses frozen observation normalization and reports raw rewards. Always keep a
policy with its matching normalization file.

## Results and checkpoint selection

Evaluation runs before training, periodically at rollout boundaries, and after
normal completion. Defaults are five fixed evaluation seeds and an evaluation
interval of 100 training episodes. These evaluations are separate from training.

Policies are ranked by highest mean success rate, then mean forward distance,
then mean raw reward. The best checkpoint can remain the untrained baseline if
training never improves those metrics. Video evaluation uses a separate seed
range by default.

| Artifact in the run folder | Contents |
| --- | --- |
| `best.zip`, `best_normalization.pkl` | Best evaluated policy and matching statistics |
| `latest.zip`, `latest_normalization.pkl` | Most recently saved policy and matching statistics |
| `best_metrics.json` | Evaluation that selected the best checkpoint |
| `evaluations.jsonl` | Evaluation summaries and individual episode results |
| `episodes.csv` | Training episode length, distance, outcome, and return |
| `reward_terms.jsonl` | Component-by-component reward totals for training episodes |
| `run.json` | Requested/completed episodes, actual training steps, update count, status, package versions |
| `config.py`, `config.json` | Selected Python settings, when supplied, and resolved settings snapshot |
| `model.xml`, `rewards.py`, `source/` | Model, reward implementation, and source snapshots |
| `best_run.mp4`, `best_run.json` | Exported video and its candidate-selection report |

Generated snapshots are not another settings file to maintain. Evaluation loads
the run's saved config, model and reward code. It still uses the installed
`environment.py`; `source/` is an audit snapshot, not an automatically executed
copy. Keep existing run snapshots intact when making other experiments.

Default `runs/` artifacts are ignored by Git; video files are also ignored by
the repository rules. Back up valuable runs separately.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| `No module named Training_run` | Run from the repository root using `python -m Training_run ...`. |
| `FileExistsError` for the run folder | Choose an unused `--run` name, or omit it for a timestamped folder. |
| Missing package / import error | Activate `.venv` and install the root `requirements.txt`. |
| MuJoCo reports Intel Python on Apple Silicon | Recreate the environment with a native ARM interpreter. |
| No PPO update occurred | The episode budget was too short to complete a rollout; increase it. |
| Video still shows the same behavior after editing rewards | Edits affect future training, not the weights of an already-saved policy. |
| Graphics-context error during export | Run video generation in a local session with working graphics access. |
| Model input-size mismatch | Use a policy, config and normalization from the same run; do not mix snapshots. |

## Code layout and checks

| File | Responsibility |
| --- | --- |
| `config.py` | User-editable settings |
| `settings.py` | Load and validate settings |
| `environment.py` | MuJoCo dynamics, observations, reset and episode rules |
| `rewards.py` | Reward formulas; edit only to introduce a different formula |
| `train.py` | PPO, episode counting, logging and checkpoint saves |
| `evaluation.py` | Policy evaluation, ranking and video export |
| `__main__.py` | Command-line interface |
| `test_running.py` | Reward, environment and training-pipeline checks |

```bash
python -m Training_run train --help
python -m Training_run video --help
python -m unittest Training_run.test_running
```

Tests check contact/finish precedence, exact episode counting, real PPO updates,
control mapping, rewards, and forward-reach gates. Passing tests verifies the
implementation—not that the agent has learned to run reliably.
