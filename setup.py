# -*- coding: utf-8 -*-


from setuptools import setup

setup(name='sfa',
      version='0.0.1',
      description='Signal flow analysis',
      url='http://github.com/dwgoon/sfa',
      author='Daewon Lee',
      author_email='daewon4you@gmail.com',
      license='MIT',
      packages=['sfa'],
      install_requires=[
          'numpy',
          'networkx',
      ],
      zip_safe=False,)
