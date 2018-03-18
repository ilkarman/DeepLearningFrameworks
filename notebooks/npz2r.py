# Function to load imdb.npz. Called by npz2r.R to convert npz to Rds

import numpy

def get_data(dataset):
  with numpy.load('imdb_processed.npz') as f:
    if dataset == "x_train":
      x_train, y_train = f['x_train'], f['y_train']
      return x_train
    elif dataset == "y_train":
      x_train, y_train = f['x_train'], f['y_train']
      return y_train
    elif dataset == "x_test":
      x_test, y_test = f['x_test'], f['y_test']
      return x_test
    elif dataset == "y_test":
      x_test, y_test = f['x_test'], f['y_test']
      return y_test
