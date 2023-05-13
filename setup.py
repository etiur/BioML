from setuptools import setup, find_packages
import BioML

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="BioML", author="Ruite Xiang", author_email="ruite.xiang@bsc.es",
      description="Automatic machine learning for the classification of protein properties",
      url="https://github.com/etiur/BioML.git", license="MIT",
      version="%s" % BioML.__version__,
      packages=find_packages(), python_requires="==3.10", long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=["Programming Language :: Python :: 3.10",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: Unix",
                   "Intended Audience :: Science/Research",
                   "Natural Language :: English",
                   "Environment :: Console",
                   "Topic :: Scientific/Engineering :: Bio-Informatics"],
      install_requires=["openpyxl", "hpsklearn@git+https://github.com/hyperopt/hyperopt-sklearn.git",
                        "scikit-learn", "joblib", "numpy", "pandas", "matplotlib", "biopython", "pyod","scipy", "combo",
                        "ITMO_FS@git+https://github.com/ctlab/ITMO_FS.git", "xgboost", "shap", "lightgbm"],
      keywords="bioprospecting, bioinformatics, machine learning")
