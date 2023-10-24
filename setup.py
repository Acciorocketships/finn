from setuptools import setup
from setuptools import find_packages

setup(
    name="finn",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch", "sympy", "sympytorch", "dill", "matplotlib"],
    author="Ryan Kortvelesy",
    author_email="rk627@cam.ac.uk",
    description="A Fixed Integral Neural Network",
)
