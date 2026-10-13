"""One-shot finalization after the existing r58 worker exits; never launch FEM."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    subprocess.run([sys.executable, '-B', str(HERE / 'analyze.py'),
                    '--wait-pid', '1436152'], check=True)
    subprocess.run([sys.executable, '-B', str(HERE / 'navigation.py')], check=True)
