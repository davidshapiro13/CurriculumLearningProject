"""PPO and experiment bookkeeping. Tune config.py instead."""
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

from .environment import RunningEnv, stock_model_path
from .evaluation import evaluate, summarize, policy_score, saved_reward
from .settings import load_config, DEFAULT_CONFIG


def train(episodes=None, config_path=DEFAULT_CONFIG, run=None):
    config = load_config(config_path)
    episodes = config['training'].get('episodes', 10000) if episodes is None else episodes
    if episodes <= 0:
        raise ValueError('episodes must be positive')
    import gymnasium
    import mujoco
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import torch

    run = Path(run or Path(__file__).parent / 'runs' / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    run.mkdir(parents=True, exist_ok=False)
    (run / 'config.json').write_text(json.dumps(config, indent=2))
    if Path(config_path).suffix == '.py':
        shutil.copyfile(config_path, run / 'config.py')
    shutil.copyfile(stock_model_path(), run / 'model.xml')
    shutil.copyfile(Path(__file__).with_name('rewards.py'), run / 'rewards.py')
    source = run / 'source'
    source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(path, source / path.name)
    manifest = dict(requested_episodes=episodes, completed_episodes=0, training_steps=0,
                    versions=dict(gymnasium=gymnasium.__version__, mujoco=mujoco.__version__,
                                  stable_baselines3=stable_baselines3.__version__, torch=torch.__version__))
    reward_fn = saved_reward(run)
    model_path = run / 'model.xml'
    torch.set_num_threads(config['training']['torch_threads'])
    ppo = dict(config['ppo'])
    hidden = ppo.pop('hidden_layers')
    initial_std = ppo.pop('log_std_init')
    activation = {'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh}[ppo.pop('activation', 'tanh')]
    ortho_init = ppo.pop('ortho_init', True)
    env = VecNormalize(DummyVecEnv([lambda: RunningEnv(config, model_path, reward_function=reward_fn)]),
                       gamma=ppo['gamma'])
    policy = PPO('MlpPolicy', env, seed=config['training']['seed'], device='cpu', verbose=0,
                 policy_kwargs=dict(net_arch=dict(pi=hidden, vf=hidden), log_std_init=initial_std,
                                    activation_fn=activation, ortho_init=ortho_init), **ppo)

    class EpisodeCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.completed = 0
            self.episode_return = 0.0
            self.best_score = None
            self.next_evaluation = config['training']['evaluate_every_episodes']
            self.started = time.monotonic()
            self.last_print = self.started
            self.last_eval_steps = -1
            self.term_totals = {}

        def save_pair(self, name):
            self.model.save(run / name)
            env.save(str(run / f'{name}_normalization.pkl'))

        def assess(self):
            if self.last_eval_steps == self.num_timesteps:
                return
            records = evaluate(self.model, env, config, model_path, reward_fn,
                               config['training']['evaluation_episodes'], config['training']['evaluation_seed'])
            metrics = summarize(records)
            record = dict(training_episodes=self.completed, training_steps=self.num_timesteps,
                          metrics=metrics, episodes=records)
            with (run / 'evaluations.jsonl').open('a') as output:
                output.write(json.dumps(record) + '\n')
            score = policy_score(metrics)
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.save_pair('best')
                (run / 'best_metrics.json').write_text(json.dumps(record, indent=2))
            self.save_pair('latest')
            self.last_eval_steps = self.num_timesteps
            print(f"Evaluation at episode {self.completed}: success {metrics['success']:.0%}, "
                  f"distance {metrics['distance']:.2f} m, return {metrics['reward']:.1f}", flush=True)

        def _on_training_start(self):
            self.assess()  # Untrained baseline also ensures a video-able checkpoint exists.

        def _on_step(self):
            self.episode_return += float(env.get_original_reward()[0])
            info = self.locals['infos'][0]
            for key, value in info['reward_terms'].items():
                self.term_totals[key] = self.term_totals.get(key, 0.0) + value
            if self.locals['dones'][0]:
                self.completed += 1
                record = dict(episode=self.completed, steps=info['steps'], distance=info['distance'],
                              success=info['success'], end_reason=info['end_reason'], reward=self.episode_return)
                path = run / 'episodes.csv'
                exists = path.exists()
                with path.open('a', newline='') as output:
                    writer = csv.DictWriter(output, fieldnames=list(record))
                    if not exists:
                        writer.writeheader()
                    writer.writerow(record)
                with (run / 'reward_terms.jsonl').open('a') as output:
                    output.write(json.dumps(dict(episode=self.completed, terms=self.term_totals)) + '\n')
                self.episode_return = 0.0
                self.term_totals = {}
            now = time.monotonic()
            if now - self.last_print >= 15:
                elapsed = now - self.started
                eta = elapsed / self.completed * (episodes - self.completed) / 60 if self.completed else None
                print(f'Episodes {self.completed}/{episodes} | steps {self.num_timesteps:,} | '
                      f'ETA {eta:.1f} min (approximate)' if eta is not None else 'Collecting first episode...', flush=True)
                self.last_print = now
            return self.completed < episodes

        def _on_rollout_end(self):
            if self.completed >= self.next_evaluation:
                self.assess()
                self.next_evaluation = self.completed + config['training']['evaluate_every_episodes']

    callback = EpisodeCallback()
    status = 'completed'
    print(f'Run: {run}\nTraining for exactly {episodes} completed episodes.', flush=True)
    try:
        policy.learn(total_timesteps=episodes * config['environment']['max_episode_steps'] + ppo['n_steps'], callback=callback)
        callback.assess()
    except KeyboardInterrupt:
        status = 'interrupted'
        print('Interrupted; saving latest checkpoint.', flush=True)
    except Exception:
        status = 'failed'
        raise
    finally:
        policy.save(run / 'latest')
        env.save(str(run / 'latest_normalization.pkl'))
        manifest.update(completed_episodes=callback.completed, training_steps=policy.num_timesteps,
                        optimizer_epochs=policy._n_updates, status=status)
        (run / 'run.json').write_text(json.dumps(manifest, indent=2))
        env.close()
    if not policy._n_updates:
        print('No PPO update occurred: use more episodes or reduce ppo.n_steps for a smoke test.', flush=True)
    print(f'Saved {run}. Completed {callback.completed} training episodes.', flush=True)
    return run
