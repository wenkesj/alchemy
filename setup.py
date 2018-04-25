# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


extras_require = {'contrib': ['pynput', 'vizdoom', 'gym', 'scipy']}
install_requires = ['tensorflow', 'numpy']
tests_require = ['gym', 'dm-sonnet']


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
