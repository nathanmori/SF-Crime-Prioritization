# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sys import argv
import os


def get_path():
    """Get the path of the directory for the project.

    INPUT
    -----
    None

    OUTPUT
    ------
    root : str
        Path of the directory for the project.
    """

    cwd = os.getcwd()
    dirs = cwd.split('/')
    n = next(i for i, dir_i in enumerate(dirs, start=1)
             if dir_i == 'ProCogia Challenge')
    path = '/'.join(dirs[:n])

    return path


def get_time_str():
    """Get the local time as a string.

    INPUT
    -----
    None

    OUTPUT
    ------
    time_str : str
        Local time as YYYYMMDDHHMMSS."""

    t = time.localtime()
    time_str = (str(t.tm_year).zfill(4) +
                str(t.tm_mon).zfill(2) +
                str(t.tm_mday).zfill(2) +
                str(t.tm_hour).zfill(2) +
                str(t.tm_min).zfill(2) +
                str(t.tm_sec).zfill(2))

    return time_str


def print_summary(df):
    """Print summary of data.

    INPUT
    -----
    df : pandas.DataFrame
        Data to summarize.

    OUTPUT
    ------
    None"""

    print 'Shape:', df.shape
    print '\nInfo:\n', df.info()
    print '\nHead:\n', df.head()
    print '\nStatistics:\n', df.describe()
    print '\nAny Nulls:\n', df.isnull().any()
    print '\nDayOfWeek value_counts:\n', df['DayOfWeek'].value_counts()
    print '\nPdDistrict value_counts:\n', df['PdDistrict'].value_counts()
    print "\nAddress has '/' or 'Block of' value_counts:\n", \
        df['Address'].apply(lambda x: '/' in x or
                            'Block of' in x).value_counts()


def get_priorities(categories):
    """Get priorities of crime categories in train data.

    General priority levels:
        1 = No harm to people or property
        2 = Minor harm to people or property
        3 = Moderate harm to people or property
        4 = Major harm to people or property

    INPUT
    -----
    categories : pandas.Series of str
        Categories of crime from train data.

    OUTPUT
    ------
    priorities : pandas.Series of int
        Priorities of crime from train data."""

    category_map = {'ARSON': 4,
                    'ASSAULT': 4,
                    'BAD CHECKS': 2,
                    'BRIBERY': 3,
                    'BURGLARY': 3,
                    'DISORDERLY CONDUCT': 2,
                    'DRIVING UNDER THE INFLUENCE': 3,
                    'DRUG/NARCOTIC': 2,
                    'DRUNKENNESS': 2,
                    'EMBEZZLEMENT': 3,
                    'EXTORTION': 3,
                    'FAMILY OFFENSES': 3,
                    'FORGERY/COUNTERFEITING': 3,
                    'FRAUD': 3,
                    'GAMBLING': 1,
                    'KIDNAPPING': 4,
                    'LARCENY/THEFT': 3,
                    'LIQUOR LAWS': 1,
                    'LOITERING': 1,
                    'MISSING PERSON': 3,
                    'NON-CRIMINAL': 1,
                    'OTHER OFFENSES': 2,
                    'PORNOGRAPHY/OBSCENE MAT': 1,
                    'PROSTITUTION': 2,
                    'RECOVERED VEHICLE': 1,
                    'ROBBERY': 3,
                    'RUNAWAY': 1,
                    'SECONDARY CODES': 2,
                    'SEX OFFENSES FORCIBLE': 4,
                    'SEX OFFENSES NON FORCIBLE': 3,
                    'STOLEN PROPERTY': 3,
                    'SUICIDE': 4,
                    'SUSPICIOUS OCC': 2,
                    'TREA': 2,
                    # Trespassing or loitering near posted industrial property
                    'TRESPASS': 2,
                    'VANDALISM': 2,
                    'VEHICLE THEFT': 3,
                    'WARRANTS': 3,
                    'WEAPON LAWS': 3}

    priorities = categories.apply(lambda x: category_map[x])

    return priorities


def get_data(shard=False, n=500, drop_category=False):
    """Get train and test data from csv files and clean.

    INPUT
    -----
    shard : bool, default=False
        Indicates if train data is limited to n rows.
    n : int, default=500
        Max number of rows of train data to read if shard=True.
    drop_category : bool, default=False
        Indicates if Category column is dropped from train data.

    OUTPUT
    ------
    df_train : pandas.DataFrame
        Train data (Priority labels included).
    df_test : pandas.DataFrame
        Test data (Priority labels not included)."""

    path = get_path()
    df_train = pd.read_csv('%s/data/train.csv' % path,
                           nrows=(n if shard else None))
    df_test = pd.read_csv('%s/data/test.csv' % path,
                          nrows=(n if shard else None))

    df_train.rename(columns={'X': 'Longitude', 'Y': 'Latitude'}, inplace=True)
    df_test.rename(columns={'X': 'Longitude', 'Y': 'Latitude'}, inplace=True)

    df_train['Priority'] = get_priorities(df_train['Category'])

    if drop_category:
        df_train.drop('Category', axis=1, inplace=True)

    # Drop Descript and Resolution columns from train data, as they are not
    # available for test data, and we are not trying to predict these values.
    df_train.drop(['Descript', 'Resolution'], axis=1, inplace=True)
    # Drop Id column from test data, as it is redundant of index
    df_test.drop('Id', axis=1, inplace=True)

    return df_train, df_test


if __name__ == '__main__':

    time_str = get_time_str()
    df_train, df_test = get_data()

    print 'TRAIN DATA\n----------\n', print_summary(df_train)
    print '\n\nTEST DATA\n---------\n', print_summary(df_test)

    print '\n\nCategories:\n', df_train['Category'].unique()
    print '\n# Categories:\n', len(df_train['Category'].unique())
    print '\nCategory value_counts:\n', df_train['Category'].value_counts()
    print '\nPriority value_counts:\n', df_train['Priority'].value_counts()
