import pandas as pd
import numpy as np
import scipy as sp
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import pdb
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.metrics import matthews_corrcoef
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score


def random_forest(df):
    print('number of total users is:', df.shape)
    print('number of unengaged is:', df[df.churn ==0].shape[0])
    df['date'] = pd.to_datetime(df.Date_out).values.astype('float')

# use only most important features
    df = df[['dtime_lb', 'dtime_pay', 'tot_spent', 'pool_users', 'com_pool_users',
         'user_referrals_count', 'pool_category', 'user_wallet', 'churn']]

# get dummy variables for pool category data
    df = pd.get_dummies(df)
    df = df[df['pool_users'] > 0]

# normalize common pool users by the tot. number of IDs in 
# the first 2 pools 
    df['com_pool_users'] /= df['pool_users']

# rename columns to make heat map corr. matrix
    df = df.rename(columns ={'dtime_lb': 'A',
                             'dtime_pay': 'B',
                             'tot_spent': 'C', 
                             'pool_users': 'D',
                             'com_pool_users': 'E', 
                             'user_referrals_count': 'F',
                             'user_wallet': 'G',
                             'churn': 'Long-Term',
                             'pool_category_charity': 'H',
                             'pool_category_contest': 'I',
                             'pool_category_gift': 'L',
                             'pool_category_other': 'M',
                             'pool_category_reunion': 'N',
                             'pool_category_sport': 'O',
                             'pool_category_trip': 'P'})

# make a copy of dataframe to be used for plotting purposes
    df_bar = df
    target = df['Long-Term']
    num_ch = len(target[target == 0])
    num_noch = len(target[target == 1])
    del df['Long-Term']
    n_features = df.shape[1]

# normalize variables
    df = pd.DataFrame(preprocessing.scale(df))

# cross validation: divide the dataset in train / test data
    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size = 0.3, random_state = 0)

# consider multiple ML algorithms
# SVC
#svc = svm.SVC()
#svc.fit(x_train, y_train)
#svc_score = svc.score(x_test, y_test)
#print('svc_score: ', svc_score)
#pdb.set_trace()

# KNN
#knn = KNeighborsClassifier(n_neighbors = 3)
#knn.fit(x_train, x_train)
#knn_score = knn.score(x_test, y_test)
#print('knn_score: ', knn_score)
#pdb.set_trace()

# Gaussian Naive Bayes
#gaussian = GaussianNB()
#gaussian.fit(x_train, y_train)
#gaussian_score = gaussian.score(x_test, y_test)
#print('gaussian_score: ', gaussian_score)
#pdb.set_trace()

# Perceptron
#perceptron = Perceptron()
#perceptron.fit(x_train, y_train)
#perceptron_score = perceptron.score(x_test, y_test)
#print('perceptron_score: ', perceptron_score)

# Decision Tree
#decision_tree = tree.DecisionTreeClassifier()
#decision_tree.fit(x_train, y_train)
#decision_tree_score = decision_tree.score(x_test, y_test)
#print('decision tree score: ', decision_tree_score)
#pdb.set_trace()

# logistic regression
#clf = linear_model.LogisticRegression()
#clf.fit(x_train, y_train)
#score = clf.score(x_test, y_test)
#print('logistic regression score: ', score)

# Random Forest with balanced classes (our categories are umbalanced)
    clf = RandomForestClassifier(n_estimators=100, class_weight = 'balanced')

# use a full grid over all parameters (not used)
#param_grid = {'n_estimators': [20, 100, 200, 500], 
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"]}
# run grid search
#grid_search = GridSearchCV(clf, param_grid=param_grid)

    clf.fit(x_train, y_train)
    predict_proba =  clf.predict_proba(x_test)
    predict = clf.predict(x_test)
# compute confusion matrix (normalized)
    cm = confusion_matrix(y_test, predict)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('matrix for feature engineered model')
    print(cm)
# plot heatmap of confusion matrix
    sns.heatmap(cm, annot = True, square=True)
    plt.show()
# compute ROC curve
    fpr, tpr, threshold = metrics.roc_curve(y_test, predict_proba[:,1], pos_label = True)
    roc_auc = metrics.auc(fpr, tpr)

# plot ROC curve
    plt.title('Receiver operating characteristic curve', fontsize=17)
    plt.plot(fpr, tpr, 'blue', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=17)
    plt.xlabel('False Positive Rate',  fontsize=17)
    plt.show()
#grid_search.fit(x_train, y_train)
#grid_search_score = grid_search.score(x_test, y_test)

    random_forest_score = clf.score(x_test, y_test)
    print('random_forest_score in the feature engineering case: ', random_forest_score)

# print/plot feature importance
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(df.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# rename variables for plotting purposes
    df_bar = df_bar[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P']]
    df_bar = df_bar.rename(columns ={'A': 'Time between invitations',
                                     'B': 'Time between payments',
                                     'C': 'Amount spent',
                                     'D': '# of pool IDs',
                                     'E': '(Common / tot) pool IDs',
                                     'F': 'User referrals count',
                                     'G': 'User wallet',
                                     'H': 'Category: Charity',
                                     'H': 'Category: Contest',
                                     'I': 'Category: Contest',
                                     'L': 'Category: Gift',
                                     'M': 'Category: Other',
                                     'N': 'Category: Reunion',
                                     'O': 'Category: Sport',
                                     'P': 'Category: Trip'})
                 
    names = df_bar.columns
    importances, names = zip(*sorted(zip(importances, names)))

    plt.barh(range(len(names)), importances, align = 'center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Percentages', fontsize=17)
    plt.ylabel('Features', fontsize=17)
    plt.title('Importance of each feature', fontsize=17)
    plt.show()

# compute precision, recall, fscore 
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, clf.predict(x_test), average = 'weighted')
    print(precision, recall, fscore, support)
