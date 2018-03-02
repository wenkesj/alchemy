# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


extras_require = {'demo': ['pynput'], 'envs': ['vizdoom', 'gym', 'scipy']}
install_requires = ['tensorflow', 'numpy']
tests_require = ['gym']


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
  extras_require=extras_require,
  tests_require=tests_require,
  platforms='any',
)
