import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score 
from sklearn.utils import shuffle

'''
  NOTES:
  B - harmless
  M - cancerous
'''

# FEATURE SELECTION
def remove_correlated_features(X):
  pass

def remove_features(X, Y):
  pass

# MODEL TRAINING
def compute_cost(W, X, Y):
  # CALCULATE HINGE LOSS
  N = X.shape[0]
  distances = 1 - Y * (np.dot(X, W))
  distances[distances < 0] = 0
  hinge_loss = reg_strenght * (np.sum(distances) / N)

  # CALCULATE LOSS
  cost = 1 / 2 * np.dot(W, W) + hinge_loss
  return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
  pass

def sgd(features, outputs):
  pass

def init():
  data = pd.read_csv('./data.csv')

  # CONVERT TO NUMERICAL VALUES
  diagnosis_map = {'M' : 1, 'B' : -1}
  data['diagnosis'] = data['diagnosis'].map(diagnosis_map)
  
  # STORE FEATURES AND OUTPUTS IN DIFFERENT DATAFRAMES
  Y = data.loc[:, 'diagnosis'] # ALL 'DIAGNOSIS' ROWS
  X = data.iloc[:, 1:] # ALL (FEATURES) ROWS OF COLUMN 1 TO END

  # NORMALIZE THE FEATURES WITH MINMAXSCALAR
  X_norm = MinMaxScaler().fit_transform(X.values)
  X = pd.DataFrame(X_norm)

  # SPLITTING THE DATASET INTO "TRAIN" AND "TEST" DATASET
  # INSERT 1 IN EVERY ROW FOR INTERCEPT B
  X.insert(loc = len(X.columns), column = 'intercept', value = 1)
  X_train, X_test, y_train, y_test = tts(X, Y, test_size = 0.2, random_state = 42)

init()