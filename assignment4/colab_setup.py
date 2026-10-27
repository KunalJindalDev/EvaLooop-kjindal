# Replacement for the failed setup cell. Run after the ZIP-upload cell.
import os
import subprocess
import sys
from pathlib import Path

os.chdir('/content/evaloop_assignment4')

def checked(command):
    print('+', ' '.join(command), flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Setup failed; copy the installation output printed above.')

# uv supplies Python itself, without relying on Colab's venv/ensurepip.
UV_DIR = '/content/evaloop_uv'
checked([sys.executable, '-m', 'pip', 'install', '--target', UV_DIR, 'uv==0.6.17'])
UV = UV_DIR + '/bin/uv'
ENV = '/content/evaloop_py312'
PYTHON = ENV + '/bin/python'
if not Path(PYTHON).exists():
    checked([UV, 'venv', '--python', '3.12.9', '--seed', ENV])
checked([PYTHON, '-m', 'pip', 'install', 'torch==2.6.0',
         '--index-url', 'https://download.pytorch.org/whl/cu124'])
checked([PYTHON, '-m', 'pip', 'install', '-r', 'assignment4/requirements-colab.txt'])
checked([PYTHON, '-m', 'pytest', 'tests/test_assignment4.py', '-q'])
checked([PYTHON, '-c',
         'import sys, torch; '
         'from evaloop.evaluation.py_testing import EvalPlusCodeTester; '
         'from transformers import Qwen2ForCausalLM; '
         'print(sys.version); print(torch.__version__); '
         'assert torch.cuda.is_available(), "Select a GPU runtime in Colab"; '
         'print(torch.cuda.get_device_name(0))'])
print('Setup complete. Continue with the next notebook cell.')
