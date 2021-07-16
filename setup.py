#!/usr/bin/env python3

import sys

try:
    from setuptools import setup
    have_setuptools = True
except ImportError:
    from distutils.core import setup
    have_setuptools = False

setup_kwargs = {
'name': 'floatcyl',
'version': '0.1',
'description': 'Series solutions for floating cylinders',
'author': 'Marco Gambarini',
'author_email': 'marco.gambarini@mail.polimi.it',
'classifiers': [
    'Programming Language :: Python :: 3',
],
'packages': ['floatcyl'],
'package_dir': {
    'floatcyl': 'floatcyl',
},
}

if __name__ == '__main__':
    setup(**setup_kwargs)
