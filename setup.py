#!/usr/bin/env python

from distutils.core import setup

setup(name='Signle Shot MultiBox Detector-SSD Fascade Parsing',
      version='1.0',
      description='Signle Shot MultiBox Detector Project',
      author='Gjorgjiev Ilija',
      author_email='ilija.gjorgjiev@epfl.ch',
      url='github.com/ilijagjorgjiev',
      packages=['ssd_project'],
      requires=[
          'tqdm',
          'torch',
          'cv2',
          'PIL',
          'numpy',
          'argparse',
          'matplotlib',
          'torchvision',
          'imageio',
          'os',
          'scipy',
          'pandas',
          'itertools',
          'glob',
          'math'
          
      ])