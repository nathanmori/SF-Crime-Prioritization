# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from eda import *
from feat_engr import FeatureEngineer
from pandas.tools.plotting import scatter_matrix
from statsmodels.graphics.mosaicplot import mosaic
from collections import OrderedDict


def priority_mosaic(df, column, show=True, save=False):
    """Create a mosaic plot of Priority by a cateogical column.

    INPUT
    -----
    df : pandas.DataFrame
        Data to plot.
    column : str
        Name of column to plot priority against.
    show : bool, default=True
        Indicates if plot is to be shown.
    save : bool, default=False
        Indicates if plot is to be saved.

    OUTPUT
    ------
    None"""

    priority_colors = {'1': 'green',
                       '2': 'orange',
                       '3': 'orangered',
                       '4': 'red'}

    ct = pd.crosstab(index=df_train['Priority'],
                     columns=df_train[column])

    crosstab_dict = OrderedDict()
    for priority, category_count in ct.iterrows():
        for category, count in category_count.iteritems():
            crosstab_dict[(category, priority)] = count

    fig, ax = plt.subplots(figsize=(12, 12))
    mosaic(crosstab_dict, ax=ax,
           properties=(lambda (category, priority):
                       {'color': priority_colors[priority]}))
    fig.suptitle('Priority By %s' % column)
    ax.set_xlabel(column)
    ax.set_ylabel('Priority')

    if save:
        plt.savefig('%s/img/%s_priority_mosaic_%s' % (path, time_str, column))

    if show:
        plt.show()
    else:
        plt.close()


def priority_violin(df, column, show=True, save=False):
    """Create a violin plot of a numerical column by Priority.

    INPUT
    -----
    df : pandas.DataFrame
        Data to plot.
    column : str
        Name of column to plot against priority.
    show : bool, default=True
        Indicates if plot is to be shown.
    save : bool, default=False
        Indicates if plot is to be saved.

    OUTPUT
    ------
    None"""

    priorities = [1, 2, 3, 4]
    data = [df[df['Priority'] == priority][column] for priority in priorities]

    plt.figure(figsize=(12, 12))
    plt.violinplot(data, priorities, showmeans=True, showextrema=True,
                   showmedians=True, bw_method=0.1)
    plt.title('%s By Priority' % column)
    plt.xlabel('Priority')
    plt.ylabel(column)
    plt.xticks([1, 2, 3, 4])

    if save:
        plt.savefig('%s/img/%s_priority_violin_%s' % (path, time_str, column))

    if show:
        plt.show()
    else:
        plt.close()


def histograms(df_train, df_test, show=True, save=False):
    """Create histograms of train and test data.

    INPUT
    -----
    df_train : pandas.DataFrame
        Train data.
    df_test : pandas.DataFrame
        Test data.
    show : bool, default=True
        Indicates if plots are to be shown.
    save : bool, default=False
        Indicates if plots are to be saved.

    OUTPUT
    ------
    None"""

    for df, title in [(df_train, 'Train'), (df_test, 'Test')]:
        df.hist(figsize=(12, 12), xrot=90, bins=7)
        plt.suptitle('%s Data Histograms' % title)

        if save:
            plt.savefig('%s/img/%s_histograms_%s' % (path, time_str, title))

        if show:
            plt.show()
        else:
            plt.close()


def samp_scat(df, frac=0.1, show=True, save=False):
    """Plot a scatter matrix of the data.

    INPUT
    -----
    df : pandas.DataFrame
        Data to plot.
    frac : float, default=0.1
        Fraction of data to plot.
    show : bool, default=True
        Indicates if plot is to be shown.
    save : bool, default=False
        Indicates if plot is to be saved.

    OUTPUT
    ------
    None"""

    scatter_matrix(df.sample(frac=frac), alpha=0.2, figsize=(12, 12))
    plt.suptitle('Scatter Matrix')

    if save:
        plt.savefig('%s/img/%s_eda_scatter' % (path, time_str))

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':

    path = get_path()
    time_str = get_time_str()

    shard = 'shard' in argv
    df_train, df_test = get_data(shard=shard, n=100, drop_category=True)
    y_train = df_train.pop('Priority').values

    feng = FeatureEngineer(dummy_PdDistrict=False, drop_PdDistrict=False)
    df_train = feng.fit_transform(df_train, y_train)
    df_test = feng.transform(df_test)

    df_train['Priority'] = y_train
    priority_mosaic(df_train, 'PdDistrict', show=False, save=True)
    priority_mosaic(df_train, 'Is_Intersection', show=False, save=True)
    priority_mosaic(df_train, 'DayOfWeek', show=False, save=True)

    priority_violin(df_train, 'DayOfYear', show=False, save=True)
    priority_violin(df_train, 'TimeOfDay', show=False, save=True)
    priority_violin(df_train, 'Street1_count', show=False, save=True)
    priority_violin(df_train, 'Street2_count', show=False, save=True)
    priority_violin(df_train, 'Street1_Street2_count', show=False, save=True)

    histograms(df_train, df_test, show=False, save=True)
