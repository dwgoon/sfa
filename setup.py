# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(name='sfa',
      version='0.0.1',
      description='Signal flow analysis',
      url='http://github.com/dwgoon/sfa',
      author='Daewon Lee',
      author_email='daewon4you@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data = {'': ['*.tsv', '*.sif', '*.json'],},
      install_requires=[
      	  'six',
          'future',
          'numpy',
          'networkx',
          'pandas',
      ],
      zip_safe=False,)
