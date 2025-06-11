#!/usr/bin/env python
# coding: utf8
import setuptools
from os import path

# Open README
here = path.abspath(path.dirname(__file__))
readme_path = path.join(here, "README.rst")
with open(readme_path, "r") as f:
    readme = f.read()

setuptools.setup(
    name="wNMFx",
    version="0.4.0",
    long_description=readme,
    description="wNMFx: weighted Non-Negative matrix Factorization in jax",
    long_description_content_type="text/x-rst",
    author="JohannesBuchner",
    author_email="johannes.buchner.acad@gmx.com",
    url="https://github.com/JohannesBuchner/weighted-nmf-jax",
    license="MIT License",
    packages=["wNMFx"],
    python_requires=">=3.6",
    install_requires=["numpy>=1.13", "jax"],
    include_package_data=False,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
