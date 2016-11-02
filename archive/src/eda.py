# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pdb
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn
import time
from sys import argv


def get_time_str():
    """
    Gets a string of the local time.

    INPUT
    -----
    None

    OUTPUT
    ------
    time_str : str
        Year, Month, Day, Hour, Min, Sec of local time.
    """

    runtime = time.localtime()
    time_str = (str(runtime.tm_year) +
                str(runtime.tm_mon) +
                str(runtime.tm_mday) +
                str(runtime.tm_hour) +
                str(runtime.tm_min) +
                str(runtime.tm_sec))

    return time_str


def print_summary(df):
    """
    Prints summary info of data.

    INPUT
    -----
    df : pandas.DataFrame
        Data to be summarized.

    OUTPUT
    ------
    None
    """

    print 'Shape:', df.shape
    print '\nInfo:\n', df.info()
    print '\nHead:\n', df.head()
    print '\nStatistics:\n', df.describe()
    print '\nAny Nulls:\n', df.isnull().any()
    print '\nDayOfWeek value_counts:\n', df.DayOfWeek.value_counts()
    print '\nPdDistrict value_counts:\n', df.PdDistrict.value_counts()
    print "\nAddress has '/' or 'Block of' value_counts:\n", \
        df.Address.apply(lambda x: '/' in x or 'Block of' in x).value_counts()


def samp_scat(df, frac=0.1, show=True, save=False):
    """
    Plots a scatter matrix of a sample of the data.

    INPUT
    -----
    df : pandas.DataFrame
        Data to be plotted.
    frac : float
        Fraction of data to plot
    show : bool
        Indicates if plot is to be shown.
    save : bool
        Indicates if plot is to be saved.

    OUTPUT
    ------
    None
    """

    scatter_matrix(df.sample(frac=frac), alpha=0.2, figsize=(12, 12),
                   diagonal='kde')
    plt.suptitle('Scatter Matrix')

    if save:
        plt.savefig('../img/%s_eda_scatter' % time_str)

    if show:
        plt.show()
    else:
        plt.close('all')


def plot_lat_lng(df, show=True, save=False):
    """
    Plots points geographically.

    INPUT
    -----
    df : pandas.DataFrame
        Data to be plotted.
    show : bool
        Indicates if plot is to be shown.
    save : bool
        Indicates if plot is to be saved.
    """

    plt.figure(figsize=(12, 12))
    plt.plot(df['X'], df['Y'], marker='.', linestyle='None',
             alpha=0.1, label='Start')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Location')

    if save:
        plt.savefig('../img/%s_lat_lng' % time_str)

    if show:
        plt.show()
    else:
        plt.close('all')


def fill_missing_lat_lng(df):
    """
    Fills missing latitudes and longitudes with mean of respective PdDistrict.

    INPUT
    -----
    df : pandas.DataFrame
        Data to be filled.
    """

    fill_mask = df['Y'] == 90

    means = {}
    for PdDistrict in df['PdDistrict'].unique():
        means[PdDistrict] = \
            df[(~ fill_mask) &
               (df['PdDistrict'] == PdDistrict)][['X', 'Y']].mean()

    for i in df[fill_mask].index:
        df.loc[i, ['X', 'Y']] = means[df.loc[i, 'PdDistrict']]

    return df


if __name__ == '__main__':

    time_str = get_time_str()

    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    fill_missing_lat_lng(df_train)
    fill_missing_lat_lng(df_test)

    print 'TRAIN DATA\n----------'
    print_summary(df_train)
    print '\n\nTEST DATA\n---------'
    print_summary(df_test)

    print '\n\nCategories:\n', df_train['Category'].unique()
    print '\n# Categories:\n', len(df_train['Category'].unique())
    print '\nCategory value_counts:\n', df_train['Category'].value_counts()

    plot_lat_lng(df_train, save=True, show=False)

    plt.close('all')
