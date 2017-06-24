import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerLine2D
import pdb
import seaborn as sns
from datetime import datetime

def merge_data(num_users):
    '''
    Routine to load and merge three datasets for:
    - user IDs
    - invitations (sent and received)
    - payments
    '''
    # load user IDs data
    users = pd.read_csv('../data/users_company.csv')
    users = users.rename(columns = lambda x: 'user_' + x)
    del users['user_Unnamed: 0']
    users = users[users.user_id < num_users]
    
    # load payments ID data
    payments = pd.read_csv('../data/payments_company.csv')
    payments = payments.rename(columns = lambda x: 'pay_' + x)
    payments = payments.rename(columns={'pay_user_id': 'user_id', 'pay_receiver_id':'pool_id'})
    del payments['pay_Unnamed: 0']
    # only consider payments made to a pool
    payments = payments[payments.pay_receiver_type == 'Pool']
    # delete failed payments 
    payments = payments[payments.pay_status != 'failed']
    
    # load liabilities (= invitations)
    liabilities = pd.read_csv('../data/liabilities_company.csv')
    liabilities = liabilities.rename(columns = lambda x: 'lb_' + x)
    liabilities = liabilities.rename(columns={'lb_pool_id': 'pool_id', 'lb_user_id': 'user_id'})
    del liabilities['lb_Unnamed: 0']
    liabilities = liabilities.sort_values(['user_id', 'lb_created_at'])
    
    # load pools
    pool = pd.read_csv('../data/pools_company.csv')
    pool = pool.rename(columns = lambda x: 'pool_' + x)

    # merge users & payments and sort by user ID
    # and date of payment creation    
    usr_pay = pd.merge(users, payments, how = 'inner', on = 'user_id')
    usr_pay = usr_pay.sort_values(['user_id', 'pay_created_at'])

    # merge users + invitations + payments 
    data = pd.merge(usr_pay, liabilities, how = 'inner', on = ['user_id', 'pool_id'])
    data = data.sort_values(['user_id', 'lb_created_at'])
    return data

def add_features(data):
    '''
    Routine to compute new features using early activity 
    of each user and add pool data 
    '''
    pool = pd.read_csv('../data/pools_ordered.csv')
    pool = pool.rename(columns = lambda x: 'pool_' + x)

    # define new features for each ID
    time_feature_lb = np.zeros(len(data.user_id.unique()))
    time_feature_pay = np.zeros(len(data.user_id.unique()))
    users_pool_feature = np.zeros(len(data.user_id.unique()))
    common_users_pool_feature = np.zeros(len(data.user_id.unique()))

    j = -1 
    month_s =  60 * 60 * 24 * 30 # month in seconds
    # main loop to compute new features
    for i in data.user_id.unique():
        j += 1
        liab_creation_unique = data[data.user_id == i].drop_duplicates(subset = ['lb_created_at'])
        lb_created_first = liab_creation_unique.iloc[0]
        pay_creation_unique = data[data.user_id == i].sort_values(['pay_created_at'])
        pay_created_first = pay_creation_unique.iloc[0]
        # date of creation of the first invitation
        Time1_lb = lb_created_first.lb_created_at
        # take data creation of the first payment
        Time1_pay = pay_created_first.pay_created_at
        first_pool = data[data.pool_id == lb_created_first.pool_id] # data on the pool created with the first liability
        first_pool_users = first_pool.user_id.unique() # unique users of the pool
        if shape(liab_creation_unique)[0] == 1: # if only 1 liability created
            time_feature_lb[j] = (pd.to_datetime(max(data.lb_updated_at)) - pd.to_datetime(Time1_lb)).total_seconds() / month_s
            users_pool_feature[j] = len(first_pool_users) - 1. # number of users associated to a pool (not counting the first)
            common_users_pool_feature[j] = 0. # no common users because only 1 liability created
        else:
        # same for second payment and invitation
            lb_created_second = liab_creation_unique.iloc[1]
            Time2_lb = lb_created_second.lb_created_at
            second_pool = data[data.pool_id == lb_created_second.pool_id]
            second_pool_users = second_pool.user_id.unique()
            Delta_time_lb = (pd.to_datetime(Time2_lb) - pd.to_datetime(Time1_lb)).total_seconds()
            Delta_time_lb /= month_s
            if Delta_time_lb <= 0:
# check for bugs in computation of time difference between invitations 
                raise RuntimeError("Time difference must be positive!")
                pdb.set_trace()
            time_feature_lb[j] = Delta_time_lb
            users_pool_feature[j] = len(first_pool_users) + len(second_pool_users) - 2 # total num of invitations
            # total num of common users
            common_users_pool_feature[j] = len(list(set(first_pool_users).intersection(second_pool_users))) - 1.
            if len(data[data.user_id  == i]) == 1: # check if there is only 1 payment
                time_feature_pay[j] = (pd.to_datetime(max(data.pay_updated_at)) - pd.to_datetime(Time1_pay)).total_seconds() / month_s
            else:
                pay_data2 = pay_creation_unique.iloc[1]
                Time2_pay = pay_data2.pay_created_at
                Delta_time_pay = (pd.to_datetime(Time2_pay) - pd.to_datetime(Time1_pay)).total_seconds()
                Delta_time_pay /= month_s
                if Delta_time_pay <= 0:
                    # check for bugs in computation of time difference between payments
                    raise RuntimeError("Time difference must be positive!")
                    pdb.set_trace()
                time_feature_pay[j] = Delta_time_pay

    df_new_features = pd.DataFrame({'dtime_lb': time_feature_lb, 'dtime_pay': time_feature_pay
                                ,'pool_users': users_pool_feature, 'com_pool_users': common_users_pool_feature})
    df_new_features['user_id'] = data.user_id.unique()
    # merge with new features created
    data = pd.merge(data, df_new_features, how = 'inner', on = 'user_id')
    data = pd.merge(data, pool, how = 'inner', on = 'pool_id')
    df = data
    num = int(len(df))
    # compute total and average money spent 
    grouped = df.pay_amount.groupby(df['user_id'])
    tot_sum = grouped.sum() # total amount spent on Moneypool
    tot_mean = grouped.mean() # average amount spent on Moneypool
    money_stats = pd.DataFrame([tot_sum.values, tot_mean.values], index = ['tot_spent', 'mean_spent']).T

    money_stats['user_id'] = tot_sum.index
    df = pd.merge(df, money_stats, how = 'inner', on = 'user_id')
    df['num_trans'] = df['tot_spent'] / df['mean_spent'] # number of transactions
    df['Date_tr'] = pd.to_datetime(df.pay_created_at) # date of transation
    df['Date_in'] = pd.to_datetime(df.user_created_at) # date creation user
    df['Date_out'] = pd.to_datetime(df.user_last_sign_in_at)
    df = df.dropna(subset = ['Date_tr'])
    df['Delta_tr_in'] = df['Date_tr'] - df['Date_in']

    # sort values by user and date of transaction
    df = df.sort_values(['user_id', 'Date_tr'])
    return df

def make_labels(df):
    '''
    Routine to compute labels and output dataset to be used for 
    ML predictions
    '''

    month_ns = 1.e9 * 60 * 60 * 24 * 30 # 1 month in ns
    time = df['Delta_tr_in'].values.astype('float')
    df['time'] = time

    # data to be used for predictions: consider only
    # 2 transactions, not the total time series
    df_2m = df.groupby('user_id').head(2)
    # delete features computed using the entire time series
    del df_2m['mean_spent']
    del df_2m['tot_spent']

    grouped = df_2m.pay_amount.groupby(df['user_id'])
    tot_sum = grouped.sum()
    tot_mean = grouped.mean()
    # recompute quantities using only 2 transactions
    money_stats = pd.DataFrame([tot_sum.values, tot_mean.values], index = ['tot_spent', 'mean_spent']).T
    money_stats['user_id'] = tot_sum.index
    df_2m = pd.merge(df_2m, money_stats, how = 'inner', on = 'user_id')
    df_2m['num_trans'] = df_2m['tot_spent'] / df_2m['mean_spent']
    df_2m['Date_tr'] = pd.to_datetime(df_2m.pay_created_at)
    df_2m['Date_in'] = pd.to_datetime(df_2m.user_created_at)
    df_2m = df_2m.dropna(subset = ['Date_tr'])
    df_2m['Delta_tr_in'] = df_2m['Date_tr'] - df_2m['Date_in']

    df_2m = df_2m.dropna(subset = ['Date_tr'])
    del df['pay_created_at']

    nusers = len(df.user_id.unique())
    users = []
    # define frequency of activity for each user, 
    # maximum length of timeseries and labels (engaged vs non-engaged)
    freq_churn = np.zeros(nusers)
    max_ts = np.zeros(nusers)
    binary_churn = np.zeros(nusers)
    # make labels for churn
    j = -1
    # label user IDs (churn vs no-churn)
    unique_users = df.user_id.unique()
    for i in unique_users:
        j+=1
        data = df[df['user_id'] == i]
        users.append(i)
        if data.shape[0] > 1:
            max_ts[j] = (max(data.Delta_tr_in.values.astype('float')) / month_ns) - data.dtime_pay[df['user_id'] ==i].values[0]
            if max_ts[j] <= 0:
                raise RuntimeError("Time difference must be positive!")
            freq_churn[j] = (data.num_trans.values[0] - 2.) / max_ts[j]
            if freq_churn[j] > 0.25 and max_ts[j] > 1: # main definition of churn here
                binary_churn[j] = 1.0
            else:
                binary_churn[j] = 0.0
    users = np.array(users)
    churn_data = pd.DataFrame([users, max_ts, freq_churn, binary_churn], index=['user_id', 'max_tseries', 'freq_churn', 'churn']).T

    df_data = pd.merge(df_2m, churn_data, how = 'inner', on = 'user_id')
    df_data = df_data.drop_duplicates(subset = ['user_last_sign_in_at'])
    # write data to be used for predictions
    df_data.to_csv('data.csv')
    return df_data
