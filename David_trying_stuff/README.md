# Original Humanoid: walking and joint previews

`my_humanoid.xml` is now the **unmodified Gymnasium humanoid**: 3 abdomen,
6 hip, 2 knee, 4 shoulder, and 2 elbow rotations (17 torque motors), plus an
uncontrolled free root. It has fixed spherical feet, no ankles, and a flexible
waist. The previous model is preserved as `humanoid_20_angles.xml`.

## Train walking

```bash
python LearningMujoco/train_walking.py --timesteps 10000000
```

The default run directory is `LearningMujoco/walking_original_run`. If it exists,
continue it with `--resume`, or choose another directory with `--output`:

```bash
python LearningMujoco/train_walking.py --resume --timesteps 1000000
```

`--timesteps` means **additional policy steps**, not episodes. Each policy step
advances up to 5 physics steps of 0.003 seconds. PPO finishes complete 512-step
rollouts, so the total may slightly exceed the requested training count.
Progress prints roughly every 15 seconds with speed, ETA, and recent episode
length. The ETA estimates the training budget, not time until walking is solved.

### Episode rules

- Gravity is enabled and the torso is unsupported.
- Terminate immediately when a non-foot body touches the ground. Check every
  physics substep; positive-distance proximity contacts do not count.
- Truncate after exactly 1000 policy steps if no fall occurred.
- No height-based early termination. A fall at step 1000 takes precedence.

### What changed in training

Actions are now **17 motor commands in [-0.4, 0.4]**, not target angles.
Actuator gearing converts motor commands into joint torque. Observations use
the stock Humanoid's 348 values: joint/root state, body inertia/velocity,
actuator forces, and external contact forces.

The reward is stock-style: `1.25 * forward_COM_velocity + upright_reward
- control_cost - contact_cost`. The upright reward is 5 when the root height
is between 1 and 2 meters and no fall has occurred. Control cost is
`0.1 * sum(control**2)`; contact cost is capped at 10. Height affects reward,
not the episode stopping rule. This rewards sustained upright motion more
strongly than the previous +1 survival reward and measures whole-body center
of mass motion instead of torso motion alone.

PPO uses the tuned Humanoid-v4 starting configuration from
[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml):
512-step rollouts, 256-sample batches, five optimization epochs, learning rate
3.56987e-5, gamma 0.95, GAE lambda 0.9, reduced initial action noise, separate
256x256 ReLU policy/value networks, and observation/reward normalization.
The Zoo config budgets 10 million steps. Our contact-based termination differs
from the stock task, so these settings do not guarantee reliable walking.

### Measure results and save progress

Every 50,000 steps (change with `--eval-every`), evaluate five fixed held-out
seeds without exploration noise. `evaluations.csv` records mean reward,
episode length, forward distance in meters, and the fraction surviving all
1000 steps. A long episode can mean standing still; inspect distance and video
as well. Best checkpoints are chosen by mean evaluation reward.

Run outputs:

- `policy.zip` and `policy_normalization.pkl`: latest matching checkpoint pair.
- `best.zip` and `best_normalization.pkl`: best matching evaluation pair.
- `best_metrics.json`, `evaluations.csv`, and episode monitor CSVs.
- `model.xml`: model snapshot used by this run; resume/evaluation use it.
- `config.json`: version, task settings, and hyperparameter source.

Ctrl-C saves the latest checkpoint. Resume loads the latest pair, retaining
optimizer state, timestep count, and normalization. Old 20-action checkpoints
in `walking_run` are preserved but **cannot be resumed with the 17-action model**.

Evaluate five episodes without rendering:

```bash
python LearningMujoco/train_walking.py --evaluate
```

Export one episode using the best checkpoint:

```bash
python LearningMujoco/train_walking.py --evaluate --episodes 1 --video walking_original.mp4
```

Add `--checkpoint policy` to inspect the latest checkpoint rather than the
best. Add `--output path/to/run` to use another run. Video export requires
macOS graphics access; training and headless evaluation do not.

## Joint previews

```bash
python LearningMujoco/animate_humanoid.py
python LearningMujoco/animate_humanoid.py --mode kick
python LearningMujoco/animate_humanoid.py --mode physics
python LearningMujoco/animate_humanoid.py --mode physics --free-root
```

The first two set poses directly. Physics mode uses a simple PD controller to
convert desired angles into stock motor commands, with a temporarily supported
torso unless `--free-root` is used. This PD controller is only for previews;
RL learns motor commands directly. Gravity/contact may prevent reaching limits.

`generate_humanoid.py` copies the original stock asset; it refuses to overwrite
an existing model. `generate_custom_humanoid.py` preserves the old custom-model
generator and writes `humanoid_20_angles.xml`, also without overwriting.

## Verification

```bash
python -m unittest discover -s LearningMujoco -p 'test_*.py'
```

Tests cover original-model structure, action bounds, deterministic resets,
contact pair ordering, foot-only contact, positive contact margins, actual
falls, within-action contact detection, and the exact episode time limit.
