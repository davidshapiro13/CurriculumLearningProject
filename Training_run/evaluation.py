"""Frozen-policy evaluation, checkpoint ranking, and best-episode video export."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from .environment import RunningEnv
from .settings import load_config


def saved_reward(run):
    spec = importlib.util.spec_from_file_location('saved_running_reward', Path(run) / 'rewards.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculate_reward


def evaluate(policy, normalizer, config, model_path, reward_function, episodes, seed):
    env = RunningEnv(config, model_path=model_path, reward_function=reward_function)
    records = []
    try:
        for index in range(episodes):
            obs, _ = env.reset(seed=seed + index)
            total = 0.0
            while True:
                action, _ = policy.predict(normalizer.normalize_obs(obs), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    records.append(dict(seed=seed + index, reward=total, steps=info['steps'],
                                        distance=info['distance'], success=info['success'], end_reason=info['end_reason']))
                    break
    finally:
        env.close()
    return records


def summarize(records):
    return {key: float(np.mean([r[key] for r in records])) for key in ('success', 'distance', 'reward', 'steps')}


def policy_score(metrics):
    # Finishing matters first; reward cannot outrank a higher success rate.
    return (metrics['success'], metrics['distance'], metrics['reward'])


def episode_score(record):
    return (record['success'], -record['steps'] if record['success'] else record['distance'], record['reward'])


def export_best_video(run, output=None, episodes=None, seed=None):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    run = Path(run)
    config = load_config(run / 'config.json')
    seed = config['training'].get('video_seed', 20000) if seed is None else seed
    reward_fn = saved_reward(run)
    model_path = run / 'model.xml'
    normalizer = VecNormalize.load(str(run / 'best_normalization.pkl'),
                                   DummyVecEnv([lambda: RunningEnv(config, model_path, reward_function=reward_fn)]))
    normalizer.training = False
    normalizer.norm_reward = False
    env = None
    try:
        policy = PPO.load(run / 'best.zip', device='cpu')
        records = evaluate(policy, normalizer, config, model_path, reward_fn,
                           episodes or config['training']['evaluation_episodes'], seed)
        best = max(records, key=episode_score)
        output = Path(output or run / 'best_run.mp4')
        output.parent.mkdir(parents=True, exist_ok=True)
        env = RunningEnv(config, model_path, 'rgb_array', reward_fn)
        obs, _ = env.reset(seed=best['seed'])
        with imageio.get_writer(str(output), fps=1 / (env.model.opt.timestep * env.task['frame_skip']), codec='libx264') as writer:
            while True:
                action, _ = policy.predict(normalizer.normalize_obs(obs), deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                image = Image.fromarray(env.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 640, 35), fill='black')
                draw.text((8, 10), f"Distance {info['distance']:.2f} / {env.task['finish_distance']:.1f} m | "
                          f"Step {info['steps']} | {info['end_reason'] or 'running'}", fill='white')
                writer.append_data(np.asarray(image))
                if terminated or truncated:
                    break
        report = dict(selected=best, candidates=records, checkpoint='best.zip')
        output.with_suffix('.json').write_text(json.dumps(report, indent=2))
        print(f'Saved {output}\nSelected episode: {best}', flush=True)
        return report
    finally:
        if env:
            env.close()
        normalizer.close()
