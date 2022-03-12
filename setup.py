from setuptools import setup, find_packages

setup(
    name = 'hyper-mixer-pytorch',
    packages = find_packages(),
    version = '0.1.0',
    install_requires=[
    'einops>=0.3',
    'torch>=1.6',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
)