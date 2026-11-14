"""Copy the original Gymnasium 17-motor humanoid without overwriting edits."""
from pathlib import Path
import gymnasium

if __name__ == '__main__':
    source = Path(gymnasium.__file__).parent / 'envs/mujoco/assets/humanoid.xml'
    target = Path(__file__).with_name('my_humanoid.xml')
    with target.open('x') as output:
        output.write(source.read_text())
    print(f'Created {target}')
