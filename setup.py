# coding: utf-8

"""
A Python module for creating corner plots
"""

import re
import os
import sys
from setuptools import setup

# Hackishly synchronize the version.
VERSION = re.findall(r"__version__ = \"(.*?)\"", open("scotchcorner.py").read())[0]

# 'setup.py publish' shortcut for publishing (e.g. setup.py from requests https://github.com/requests/requests/blob/master/setup.py)
# 'setup.py publish' shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel --universal')
    os.system('twine upload dist/*')
    sys.exit()

setup(
    name="scotchcorner",
    version=VERSION,
    author="Matthew Pitkin",
    author_email="matthew.pitkin@glasgow.ac.uk",
    url="https://github.com/mattpitkin/scotchcorner",
    py_modules=["scotchcorner"],
    description="A different corner plot.",
    package_data={"": ["LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
)
