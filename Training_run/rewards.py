"""Reward formulas; normal tuning belongs in config.py.

The default old_baseline uses the original per-action dense rewards.
The experimental mode preserves compatibility with earlier JSON snapshots.
"""
import numpy as np


def calculate_reward(state, weights):
    """Pure function: state measurements -> total reward and named components."""
    w = weights
    if w.get('mode') in ('per_action', 'old_baseline'):
        return old_baseline_reward(state, w)
    dt = state['dt']
    upright = max(0.0, state['torso_up']) * np.exp(
        -((state['height'] - w['standing_height']) / w['height_tolerance']) ** 2)
    # Opposite phases impose the same task on each leg, half a cycle apart.
    midpoint = np.deg2rad((w['knee_straight_degrees'] + w['knee_bent_degrees']) / 2)
    amplitude = np.deg2rad((w['knee_straight_degrees'] - w['knee_bent_degrees']) / 2)
    targets = midpoint + amplitude * np.array([np.sin(state['phase']), -np.sin(state['phase'])])
    knee_errors = (np.asarray(state['knees']) - targets) / np.deg2rad(w['knee_tolerance_degrees'])
    # Both knees must track: a straight dragging leg reduces this joint score.
    knee_match = np.exp(-np.mean(knee_errors ** 2))
    moving = np.clip(state['vx'] / w['knee_speed_gate'], 0, 1)
    terms = {
        'progress': w['progress_weight'] * state['dx'],
        'speed': dt * w['speed_weight'] * np.exp(-((state['vx'] - w['target_speed']) / w['speed_tolerance']) ** 2),
        'upright': dt * w['upright_weight'] * upright,
        'heading': dt * w['heading_weight'] * state['heading'],
        'side_velocity': -dt * w['side_velocity_cost'] * state['vy'] ** 2,
        'side_position': -dt * w['side_position_cost'] * state['lateral_distance'] ** 2,
        'knees': dt * w['knee_weight'] * knee_match * moving * upright,
        'effort': -dt * w['effort_cost'] * float(np.mean(np.square(state['action']))),
        'action_change': -w['action_change_cost'] * float(np.mean(np.square(state['action'] - state['previous_action']))),
        'fall': -w['fall_penalty'] if state['fell'] else 0.0,
        'finish': w['finish_bonus'] if state['success'] else 0.0,
        'forward_reach': forward_reach_reward(state, w),
    }
    return float(sum(terms.values())), {name: float(value) for name, value in terms.items()}


def old_baseline_reward(s, w):
    """Original dense scale and formulas, plus separately adjustable finish/fall terms."""
    height = np.exp(-((s['height'] - w['standing_height']) / w['height_tolerance']) ** 2)
    headings = s['body_headings']
    terms = {
        'forward': w['forward_velocity_weight'] * s['vx'],
        'upright': 0.0 if s['fell'] else w['upright_weight'] * height,
        'heading': w['torso_heading_weight'] * headings['torso']
                   + w['pelvis_heading_weight'] * headings['pelvis']
                   + w['foot_heading_weight'] * (headings['left_foot'] + headings['right_foot']),
        'effort': -w['effort_cost'] * float(np.sum(np.square(s['motor_controls']))),
        'contact': -min(w['contact_cost'] * s['contact_force_squared'], w['contact_cost_cap']),
        'fall': -w['fall_penalty'] if s['fell'] else 0.,
        'finish': w['finish_bonus'] if s['success'] else 0.,
    }
    # Optional experimental terms retain their previous meanings/scales. Disabled
    # by default so the old baseline isn't changed by extra objectives.
    optional = dict(w, mode='experimental')
    _, extras = calculate_reward(s, optional)
    for name in ('progress', 'speed', 'side_velocity', 'side_position', 'knees', 'action_change', 'forward_reach'):
        terms[name] = extras[name]
    return float(sum(terms.values())), {k: float(v) for k, v in terms.items()}


def forward_reach_reward(s, w):
    """Capped swing reach, subordinate to upright, forward-moving locomotion.

    Both legs receive the same formula. This encourages reach, not alternation;
    a held-out foot earns nothing once it stops advancing relative to the pelvis.
    """
    weight = w.get('forward_reach_weight', 0.0)
    if not weight or s['fell']:
        return 0.0
    min_up = w['forward_reach_min_up']
    upright = np.clip((s['torso_up'] - min_up) / (1 - min_up), 0, 1)
    height = np.exp(-((s['height'] - w['standing_height']) / w['height_tolerance']) ** 2)
    moving = np.clip(s['vx'] / w['forward_reach_speed'], 0, 1)
    heading = np.clip(s['heading'], 0, 1)
    feet = []
    for foot in s['swing_feet']:
        if foot['contact'] or foot['clearance'] < w['forward_reach_clearance']:
            feet.append(0.0)
        else:
            reach = np.clip(foot['ahead'] / w['forward_reach_meters'], 0, 1)
            advancing = np.clip(foot['relative_vx'] / w['forward_reach_swing_speed'], 0, 1)
            feet.append(float(reach * advancing))
    return float(weight * upright * height * moving * heading * np.mean(feet))
