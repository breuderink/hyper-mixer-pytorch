from setuptools import setup, find_packages

setup(
    name="hyper-mixer-pytorch",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "torch>=1.6",
    ],
)
