# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(name='sfa',
      version='0.1.0',
      description='Signal flow analysis',
      url='http://github.com/dwgoon/sfa',
      author='Daewon Lee',
      author_email='daewon4you@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={'': ['*.tsv', '*.sif', '*.json'], },
      python_requires='>=3.7',
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'networkx',
      ],
      zip_safe=False, )
