# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import time
from eda import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sys import argv
from imblearn.over_sampling import RandomOverSampler
import csv
from sklearn.preprocessing import StandardScaler


def split_address(address):
    """
    Takes one Address value and returns is_intersection, street1, street2.

    INPUT
    -----
    address : str
        Raw address text.

    OUTPUT
    ------
    is_intersection : int
        0 if address given as street1 Block of street2, 1 if street1 / street2.
    street1 : str
        First street name contained in address.
    street2 : str
        Second street name contained in address.
    """

    if 'Block of' in address:
        is_intersection = 0
        street1, street2 = address.split('Block of')
        return [is_intersection, street1, street2]

    else:
        is_intersection = 1
        street1, street2 = address.split('/')
        return [is_intersection, street1, street2]


if __name__ == '__main__':

    time_str = get_time_str()
    shard = 'shard' in argv

    # Reads train and test data from .csv files to pandas dataframes
    df_train = pd.read_csv('../data/train.csv',
                           nrows=(1000 if shard else None))
    df_test = pd.read_csv('../data/test.csv')
    dfs = [df_train, df_test]

    # Drops Descript and Resolution columns from train data, as they are not
    # available for test data, and we are not trying to predict these values
    df_train.drop(['Descript', 'Resolution'], axis=1, inplace=True)
    # Drops Id column from test data, as it is redundant of index
    df_test.drop('Id', axis=1, inplace=True)

    # Loops over train and test data and performs data preparation steps
    dfs = [df_train, df_test]
    for i, df in enumerate(dfs):

        # Fills "missing" lat and long coordinates
        fill_missing_lat_lng(df)

        # Converts Dates column to Year, Month, Weekday (as int rather than
        # str), Hour (including minutes and seconds)
        df['Dates'] = df['Dates'].apply(lambda x:
                                        datetime.strptime(x,
                                                          '%Y-%m-%d %H:%M:%S'))
        df['Year'] = df['Dates'].apply(lambda x: x.year)
        df['Month'] = df['Dates'].apply(lambda x: x.month)
        df['Weekday'] = df['Dates'].apply(lambda x: x.weekday())
        df['Hour'] = df['Dates'].apply(lambda x: x.hour + x.minute / 60 +
                                       x.second / 3600)
        df.drop(['Dates', 'DayOfWeek'], axis=1, inplace=True)

        # Converts Address to Is_Intersection, Street1, and Street2
        df_addresses = \
            pd.DataFrame(np.array(df['Address'].apply(split_address).tolist()),
                         columns=['Is_Intersection', 'Street1', 'Street2'])
        df_addresses['Is_Intersection'].apply(int)
        df = pd.concat([df, df_addresses], axis=1)
        df.drop('Address', axis=1, inplace=True)
        # Street1 and Street2 strings are categorical, and need to be modified
        # to feed into models. This info is somewhat redundant with latitude
        # and longitude, so will be dropped for this case. Could try to do more
        # with these.
        df.drop(['Street1', 'Street2'], axis=1, inplace=True)

        # Converts PdDistrict to dummy columns
        df = pd.get_dummies(df, columns=['PdDistrict'], drop_first=True)

        dfs[i] = df

    df_train, df_test = dfs

    # Converts dataframes to arrays
    y_train = df_train.pop('Category').values
    X_train = df_train.values
    X_test = df_test.values
    # Oversamples to address class imbalance
    X_resampled, y_resampled = RandomOverSampler().fit_sample(X_train, y_train)
    # Scales data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)
    # Takes a sample of the oversampled data to speed up runtime
    n = X_train.shape[0]
    n_over = X_train_scaled.shape[0]
    sample = np.random.choice(n_over, size=n, replace=False)
    X_train_sample = X_train_scaled[sample, :]
    y_train_sample = y_resampled[sample]

    mods = [LogisticRegression(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            AdaBoostClassifier(),
            SVC()]

    param_grids = {LogisticRegression:
                   {'C': [0.01, 0.1, 1.0]},
                   RandomForestClassifier:
                   {'max_features': ['auto', None],
                    'max_depth': [None, 5]},
                   GradientBoostingClassifier:
                   {'max_depth': [2, 3, 5]},
                   AdaBoostClassifier:
                   {'learning_rate': [0.01, 0.1, 1]},
                   SVC:
                   {'kernel': ['rbf', 'poly']}}

    # Searches for best parameters for each model
    grids = []
    for mod in mods:
        param_grid = param_grids[mod.__class__]
        grid = GridSearchCV(mod, param_grid=param_grid, n_jobs=-1)
        grid.fit(X_train_sample, y_train_sample)
        grids.append(grid)

    # Displays model scores and best model info
    scores = [grid.best_score_ for grid in grids]
    ix_max_score = np.argmax(scores)
    best_score = scores[ix_max_score]
    best_mod = mods[ix_max_score]
    best_grid = grids[ix_max_score]

    print '\nMODEL SCORES\n------------'
    for mod, score in zip(mods, scores):
        print '%s: %f' % (mod.__class__.__name__, score)

    print '\nBEST SCORE\n----------'
    print best_score

    print '\nBEST MODEL\n----------'
    print best_mod.__class__.__name__

    print '\nBEST MODEL PARAMS\n-----------------'
    print best_mod.get_params()

    # Uses best model to predict test data and save predictions to .csv
    y_test_pred = best_grid.predict(X_test_scaled)
    df_test['Category_Predicted'] = y_test_pred
    fname = ('../output/shard_%s_test_preds.csv' % time_str if shard else
             '../output/%s_test_preds.csv' % time_str)
    df_test.to_csv(fname,
                   columns=['Category_Predicted'],
                   header=True,
                   index_label='Id',
                   encoding='utf-8')
