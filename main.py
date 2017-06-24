'''
Main routine to:
1) load datasets for a given number of users
2) compute new features from the data and add to dataset
3) label the data (engaged vs non-engaged) to be used 
   for predictions
'''

import numpy as np
import pdb
from datetime import datetime
import alldata
import ml_algo

# number of users to be considered
num_users = int(input("How many users? "))

# load and merge IDs + invitations + payments
data = alldata.merge_data(num_users)

# compute new features and add pool data
# return dataset with new features added
data_w_feat = alldata.add_features(data)

# make labeled data and output data file
# to be used for predictions
data_labeled = alldata.make_labels(data_w_feat)

# compute ML predictions and main summary statistics
ml_algo.random_forest(data_labeled)
