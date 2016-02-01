#!/usr/bin/env python

import re
from setuptools import setup

# Hackishly synchronize the version.
version = re.findall(r"__version__ = \"(.*?)\"", open("newcorner.py").read())[0]

setup(
    name="newcorner",
    version=version,
    author="Matthew Pitkin",
    author_email="matthew.pitkin@glasgow.ac.uk",
    url="https://github.com/mattpitkin/newcorner.py",
    py_modules=["newcorner"],
    description="A different corner plot.",
    package_data={"": ["LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
)
