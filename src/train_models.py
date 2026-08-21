import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import randomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import argparse 
from xgboost import XGBClassifier

from src.preprocess_data import load_data, get_features_target, get_numerical_categorical_features, 


INPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / 'splits'
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / 'models'

