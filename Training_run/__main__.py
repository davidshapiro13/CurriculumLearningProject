"""CLI: python -m Training_run train ... / python -m Training_run video ..."""
import argparse
from pathlib import Path
from .settings import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Train and inspect the original MuJoCo Humanoid runner')
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--episodes', type=int, help='Overrides TRAIN_EPISODES in config.py')
    training.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    training.add_argument('--run', type=Path, help='New output directory; default is a timestamped runs/ folder')
    training.add_argument('--video', action='store_true', help='Export best episode after training')
    video = sub.add_parser('video')
    video.add_argument('--run', type=Path, required=True)
    video.add_argument('--output', type=Path)
    video.add_argument('--episodes', type=int, default=None, help='Overrides configured evaluation candidate count')
    video.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error('--episodes must be positive')
    if args.command == 'train':
        from .train import train
        run = train(args.episodes, args.config, args.run)
        if args.video:
            from .evaluation import export_best_video
            export_best_video(run)
    else:
        from .evaluation import export_best_video
        export_best_video(args.run, args.output, args.episodes, args.seed)


if __name__ == '__main__':
    main()
