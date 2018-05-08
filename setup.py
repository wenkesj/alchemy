# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


install_requires = ['tensorflow', 'numpy', 'dm-sonnet', 'pynput', 'gym']

setup(
  name='alchemy',
  version='0.0.1',
  long_description='',
  author='Sam Wenke',
  author_email='samwenke@gmail.com',
  license='MIT',
  description=(''),
  packages=find_packages('.'),
  install_requires=install_requires,
  platforms='any',
)
