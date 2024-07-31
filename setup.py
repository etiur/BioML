from setuptools import setup, find_packages
import BioML

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="BioML", author="Ruite Xiang", author_email="ruite.xiang@bsc.es",
      description="Automatic machine learning for the classification of protein properties",
      url="https://github.com/etiur/BioML.git", license="MIT",
      version="%s" % BioML.__version__,
      packages=find_packages(), python_requires=">=3.9", long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=["Programming Language :: Python :: 3.11",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: Unix",
                   "Intended Audience :: Science/Research",
                   "Natural Language :: English",
                   "Environment :: Console",
                   "Topic :: Scientific/Engineering :: Bio-Informatics"],
      keywords="bioprospecting, bioinformatics, machine learning")
