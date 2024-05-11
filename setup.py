#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

install_requires = [
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='winter_green_house',
      version='0.0.1',
      description='Winter Green House Modelling.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/joshuaalbert/winter_green_house",
      author='Joshua G. Albert',
      author_email='albert@strw.leidenuniv.nl',
      install_requires=install_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      package_dir={'': './'},
      packages=find_packages('./'),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.9',
      )
