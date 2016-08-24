# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().split('\n')

setup(
    name='smoomapy',
    version='0.0.1',
    author="mthh",
    author_email="matthieu.viry@ums-riate.fr",
    packages=find_packages(),
    test_suite="tests",
    install_requires=requirements
    )
