import imp
import os
import sys
from pathlib import Path

from setuptools import find_packages, setup 


sys.path.insert(1, os.path.join(sys.path[0], ".."))

with open("requirements.txt") as f:
    required = f.read().splitlines()

os.chdir(Path(__file__).parent.absolute())
setup(
    name="discretenetwork", 
    version="0.0.1",
    packages=find_packages(
        include=["discrete_network", "discrete_network.*", "network", "network.*", "device", "device.*", "method", "method.*" ]
    ),
    install_requires=required,
    tests_require=["pytest"],
)