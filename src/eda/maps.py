# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from eda import *
from PIL import Image


def map_priorities_all(df, show=True, save=False):
    """Scatter plot all crime locations by priority.

    INPUT
    -----
    df : pandas.DataFrame
        Data to plot.
    show : bool, default=True
        Indicates if plot is to be shown.
    save : bool, default=False
        Indicates if plot is to be saved.

    OUTPUT
    ------
    None"""

    colors = ['green', 'orange', 'orangered', 'red']
    markers = ['o', 'o', '.', '.']

    plt.figure(figsize=(15, 10))
    plt.imshow(sf_img, zorder=0, cmap='Greys_r',
               extent=[-122.570, -122.316, 37.683, 37.831])

    for priority, (color, marker) in enumerate(zip(colors, markers), start=1):
        mask = df['Priority'] == priority
        plt.scatter(df[mask]['Longitude'], df[mask]['Latitude'], marker=marker,
                    alpha=0.02, color=color, label='Priority %d' % priority)
    plt.xlim((-1.223*10**2 - 0.24, -1.223*10**2 - 0.06))
    plt.ylim((37.7, 37.82))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Crime Priorities By Location')

    leg = plt.legend(loc='best')
    for text in leg.get_texts():
        text.set_color('white')

    if save:
        plt.savefig('%s/img/%s_map_priority_all' % (path, time_str))

    if show:
        plt.show()
    else:
        plt.close()


def map_priorities(df, show=True, save=False):
    """Scatter plot crime locations by priority.

    INPUT
    -----
    df : pandas.DataFrame
        Data to plot.
    show : bool, default=True
        Indicates if plot is to be shown.
    save : bool, default=False
        Indicates if plot is to be saved.

    OUTPUT
    ------
    None"""

    fig, ((ax1, ax2), (ax3, ax4)) = \
        plt.subplots(2, 2, sharex='col', sharey='row', figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]

    colors = ['green', 'orange', 'orangered', 'red']

    for priority, (ax, color) in enumerate(zip(axes, colors), start=1):
        ax.imshow(sf_img, zorder=0, cmap='Greys_r',
                  extent=[-122.570, -122.316, 37.683, 37.831])

        mask = df['Priority'] == priority
        ax.scatter(df[mask]['Longitude'], df[mask]['Latitude'], marker='.',
                   alpha=0.01, color=color,
                   label='Priority %d' % priority)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Priority %d' % priority)
        ax.set_xlim((-1.223*10**2 - 0.24, -1.223*10**2 - 0.06))
        ax.set_ylim((37.7, 37.82))

    fig.suptitle('Crime Locations By Priority')

    if save:
        plt.savefig('../../img/%s_map_priority' % time_str)

    if show:
        plt.show()
    else:
        plt.close()


def map_PdDistrict(df, show=True, save=False):
    """Scatter plot crime locations by PdDistrict.

    INPUT
    -----
    df : pandas.DataFrame
        Data to plot.
    show : bool, default=True
        Indicates if plot is to be shown.
    save : bool, default=False
        Indicates if plot is to be saved.

    OUTPUT
    ------
    None"""

    districts = np.unique(df_train['PdDistrict'])
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'aqua', 'darkgreen',
              'pink']

    plt.figure(figsize=(15, 10))
    plt.imshow(sf_img, zorder=0, cmap='Greys_r',
               extent=[-122.570, -122.316, 37.683, 37.831])

    for district, color in zip(districts, colors):
        mask = df['PdDistrict'] == district
        plt.scatter(df[mask]['Longitude'], df[mask]['Latitude'], marker='o',
                    alpha=1, color=color, label=district)
    plt.xlim((-1.223*10**2 - 0.24, -1.223*10**2 - 0.06))
    plt.ylim((37.7, 37.82))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('PdDistrict By Location')

    leg = plt.legend(loc='best')
    for text in leg.get_texts():
        text.set_color('white')

    if save:
        plt.savefig('../../img/%s_map_PdDistrict' % time_str)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':

    time_str = get_time_str()
    path = get_path()
    df_train, df_test = get_data()
    sf_img = np.asarray(Image.open('%s/data/SF.png' % path).convert("L"))

    map_priorities_all(df_train, save=True, show=False)
    map_priorities(df_train, save=True, show=False)
    map_PdDistrict(df_train, save=True, show=False)
