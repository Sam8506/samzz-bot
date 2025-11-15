import sys
from pathlib import Path


print('Python:', sys.executable, sys.version)
print('Virtualenv active?' , 'venv' in sys.prefix)


data_files = list(Path('data').glob('*'))
print('Data files:', [str(p) for p in data_files])