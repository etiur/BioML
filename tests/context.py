import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import BioML.features as features
from BioML.utilities import scale
from BioML.training.classification import Classifier
from BioML.training.regression import Regressor