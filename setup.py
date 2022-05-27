from setuptools import setup, find_packages

setup(
    name="hyper-mixer-pytorch",
    version="0.1.1",
    author="Boris Reuderink",
    author_email="boris@cortext.nl",
    url="https://github.com/breuderink/hyper-mixer-pytorch",
    packages=find_packages(),
    install_requires=[
        "torch>=1.6",
    ],
)
