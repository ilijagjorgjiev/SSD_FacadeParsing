#!/usr/bin/env python
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
     name='ssd_project',  
     version='0.1' ,
     author="Ilija Gjorgjiev",
     author_email="ilija.gjorgjiev@epfl.ch",
     description="Single Shot MultiBox Detector - SSD",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/ilijagjorgjiev/SSD_FascadeParsing",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
     ],
 )
