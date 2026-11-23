"""Adjustable settings for humanoid training and evaluation.

Settings are captured when training starts. Saved runs use their own snapshots.
"""

# Episode rules and simulation.
# Distances are in meters; frame skip is physics steps per policy action.
ENV_FINISH_DISTANCE = 20.0
ENV_MAX_EPISODE_STEPS = 1000
ENV_FRAME_SKIP = 5
ENV_RESET_NOISE = 0.01
ENV_GAIT_FREQUENCY_HZ = 1.4
ENV_OBSERVATION_MODE = 'stock'
ENV_CONTROL_MODE = 'native'
ENV_RESET_STYLE = 'state_noise'  # Perturb starting positions and velocities.

# Reward weights and targets. A zero weight disables that component.
# Per-action mode does not multiply the core rewards by elapsed time.
# Optional speed, sideways, and knee terms are time-scaled.
# Fall penalties are subtracted; finish bonuses are added once per episode.
REWARD_FALL_PENALTY = 200.0
REWARD_FINISH_BONUS = 500.0
REWARD_PROGRESS_WEIGHT = 0.0
REWARD_TARGET_SPEED = 3.0
REWARD_SPEED_TOLERANCE = 1.5
REWARD_SPEED_WEIGHT = 0.0
REWARD_UPRIGHT_WEIGHT = 5.0
REWARD_STANDING_HEIGHT = 1.4
REWARD_HEIGHT_TOLERANCE = 0.15
REWARD_HEADING_WEIGHT = 1.0  # Used only in time_scaled reward mode.
REWARD_SIDE_VELOCITY_COST = 0.0
REWARD_SIDE_POSITION_COST = 0.0
REWARD_KNEE_WEIGHT = 0.0
REWARD_KNEE_STRAIGHT_DEGREES = -10.0
REWARD_KNEE_BENT_DEGREES = -65.0
REWARD_KNEE_TOLERANCE_DEGREES = 25.0
REWARD_KNEE_SPEED_GATE = 0.5
REWARD_EFFORT_COST = 0.2
REWARD_ACTION_CHANGE_COST = 0.0
REWARD_MODE = 'per_action'
REWARD_FORWARD_VELOCITY_WEIGHT = 1.25
REWARD_CONTACT_COST = 5e-07
REWARD_CONTACT_COST_CAP = 10.0
REWARD_TORSO_HEADING_WEIGHT = 0.5
REWARD_PELVIS_HEADING_WEIGHT = 0.5
REWARD_FOOT_HEADING_WEIGHT = 0.25

# Small secondary bonus for a foot reaching forward during its swing.
# Maximum total bonus per action; compare with the upright reward of 5.0.
REWARD_FORWARD_REACH_WEIGHT = 0.5
REWARD_FORWARD_REACH_METERS = 0.35  # Foot ahead of pelvis; no extra reward beyond this.
REWARD_FORWARD_REACH_CLEARANCE = 0.03  # Minimum foot-bottom height above the floor.
REWARD_FORWARD_REACH_SPEED = 0.5  # Body speed at which the movement gate reaches 1.
REWARD_FORWARD_REACH_SWING_SPEED = 1.0  # Foot speed forward relative to pelvis.
REWARD_FORWARD_REACH_MIN_UP = 0.8  # No bonus when the torso tilts too far.

# PPO learning and neural-network settings.
# N_STEPS is rollout length; N_EPOCHS is optimization passes per rollout.
PPO_LEARNING_RATE = 3.56987e-05
PPO_N_STEPS = 512
PPO_BATCH_SIZE = 256
PPO_N_EPOCHS = 5
PPO_GAMMA = 0.95
PPO_GAE_LAMBDA = 0.9
PPO_CLIP_RANGE = 0.3
PPO_ENT_COEF = 0.00238306
PPO_VF_COEF = 0.431892
PPO_MAX_GRAD_NORM = 2.0
PPO_HIDDEN_LAYERS = [256, 256]
PPO_LOG_STD_INIT = -2.0
PPO_ACTIVATION = 'relu'
PPO_ORTHO_INIT = False

# Run length and evaluation. CLI --episodes overrides TRAIN_EPISODES.
TRAIN_SEED = 0
TRAIN_TORCH_THREADS = 1
TRAIN_EVALUATE_EVERY_EPISODES = 100
TRAIN_EVALUATION_EPISODES = 5
TRAIN_EVALUATION_SEED = 10000
TRAIN_EPISODES = 10000
TRAIN_VIDEO_SEED = 20000
