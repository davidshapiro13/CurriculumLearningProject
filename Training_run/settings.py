"""Configuration loading; edit config.py rather than this plumbing."""
import json
import runpy
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name('config.py')


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if path.suffix == '.py':
        variables = runpy.run_path(str(path))
        config = {section: {key[len(prefix) + 1:].lower(): value
                           for key, value in variables.items() if key.startswith(prefix + '_')}
                  for section, prefix in {'environment': 'ENV', 'rewards': 'REWARD',
                                          'ppo': 'PPO', 'training': 'TRAIN'}.items()}
    else:
        # Existing run snapshots and older experiment JSON remain readable.
        config = json.loads(path.read_text())
    for section, keys in {
        'environment': ('finish_distance', 'max_episode_steps', 'frame_skip', 'gait_frequency_hz'),
        'rewards': ('speed_tolerance', 'height_tolerance', 'knee_tolerance_degrees', 'knee_speed_gate'),
        'ppo': ('learning_rate', 'n_steps', 'batch_size', 'n_epochs'),
        'training': ('evaluate_every_episodes', 'evaluation_episodes', 'torch_threads'),
    }.items():
        for key in keys:
            if config[section][key] <= 0:
                raise ValueError(f'{section}.{key} must be positive')
    for section, keys in {'environment': ('max_episode_steps', 'frame_skip'),
                          'ppo': ('n_steps', 'batch_size', 'n_epochs'),
                          'training': ('evaluate_every_episodes', 'evaluation_episodes', 'torch_threads')}.items():
        for key in keys:
            if not isinstance(config[section][key], int):
                raise ValueError(f'{section}.{key} must be an integer')
    ppo = config['ppo']
    rewards = config['rewards']
    if rewards.get('forward_reach_weight', 0) < 0:
        raise ValueError('forward_reach_weight must be nonnegative')
    if rewards.get('forward_reach_weight', 0):
        for key in ('forward_reach_meters', 'forward_reach_speed', 'forward_reach_swing_speed'):
            if rewards[key] <= 0:
                raise ValueError(f'rewards.{key} must be positive')
        if not 0 <= rewards['forward_reach_min_up'] < 1 or rewards['forward_reach_clearance'] < 0:
            raise ValueError('Invalid forward reach posture or clearance threshold')
    if ppo['n_steps'] < 2 or ppo['batch_size'] < 2 or ppo['n_steps'] % ppo['batch_size']:
        raise ValueError('n_steps must be a multiple of batch_size; both must be >= 2')
    if not 0 < ppo['gamma'] <= 1 or not 0 <= ppo['gae_lambda'] <= 1:
        raise ValueError('Invalid PPO discount or GAE lambda')
    return config
