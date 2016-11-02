# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import numpy as np
import pandas as pd
from eda import *
from datetime import datetime
from collections import Counter
import pdb


def split_address(address):
    """Split an Address value into is_intersection, street1, street2.

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
        Second street name contained in address."""

    if 'Block of' in address:
        is_intersection = 0
        street1, street2 = address.split('Block of')
        return [is_intersection, street1, street2]

    else:
        is_intersection = 1
        street1, street2 = address.split('/')
        return [is_intersection, street1, street2]


class UserDefinedTransform(object):
    """User-defined scikit-learn style transform class.

    Parameters
    ----------
    None"""

    def __init__(self):

        self.params = {}

    def fit(self, *args):
        """Empty scikit-learn style fit method for child classes requiring no
        fit operations. Used for Pipeline compatibility.

        INPUT
        -----
        *args : args
            Additional parameters passed to the fit function of the estimator.

        OUTPUT
        ------
        self : object
            Returns self."""

        return self

    def get_params(self, deep=True):
        """Get parameters of the estimator.

        INPUT
        -----
        deep : bool, optional
            For compatibility with scikit-learn classes.

        OUTPUT
        ------
        self.params : mapping of string to any
            Parameter names mapped to their values."""

        return self.params

    def set_params(self, **params):
        """Validate and set the parameters of the estimator.

        Parameters
        ----------
        **params : kwargs
            Parameter names mapped to the values to be set.

        Returns
        -------
        None"""

        for key, value in params.iteritems():
            self.params[key] = value


class FeatureEngineer(UserDefinedTransform):
    """Performs feature engineering.

    Parameters
    ----------
    dummy_PdDistrict : bool, default=True
        Indicates if PdDistrict is converted to dummy variables.
    drop_PdDistrict : bool, default=True
        Indicates if PdDistrict column is dropped.
    include_Mean : bool, default=True
        Indicates if PdDistrict_Mean_Priority column is included.
    include_Intersection : bool, default=True
        Indicates if Is_Intersection column is included."""

    def __init__(self, dummy_PdDistrict=True, drop_PdDistrict=True,
                 include_Mean=True, include_Intersection=True):

        self.params = {'dummy_PdDistrict': dummy_PdDistrict,
                       'drop_PdDistrict': drop_PdDistrict,
                       'include_Mean': include_Mean,
                       'include_Intersection': include_Intersection}

    def fit_transform(self, X, y):
        """Fit to data and transform.

        INPUT
        -----
        X : pandas.DataFrame
            Data from get_data, must include these columns:
            Dates, PdDistrict, Address, Latitude, Longitude
        y : numpy.array of shape [n_samples]
            Target values.

        OUTPUT
        ------
        df : pandas.DataFrame
            Data with feature engineering applied."""

        df = X.copy()

        # Convert Dates column to Year, Month, Day, Weekday (as int rather than
        # str), Hour (including minutes and seconds)
        df['Dates'] = df['Dates'].apply(lambda x:
                                        datetime.strptime(x,
                                                          '%Y-%m-%d %H:%M:%S'))
        df['DayOfYear'] = df['Dates'].apply(lambda x: x.dayofyear)
        df['DayOfWeek'] = df['Dates'].apply(lambda x: x.weekday())
        df['TimeOfDay'] = df['Dates'].apply(lambda x: x.hour + x.minute / 60. +
                                            x.second / 3600.)
        df.drop('Dates', axis=1, inplace=True)

        # Convert Address column to Is_Intersection and three counter
        # columns for the two streets contained in the address and the
        # combination of the two
        addresses = pd.DataFrame(df['Address'].apply(split_address).tolist(),
                                 columns=['Is_Intersection', 'Street1',
                                          'Street2'])
        df.drop('Address', axis=1, inplace=True)
        self.street1_counter = Counter()
        self.street2_counter = Counter()
        self.combined_counter = Counter()
        for i, row in addresses.iterrows():
            self.street1_counter[row['Street1']] += 1
            self.street2_counter[row['Street2']] += 1
            self.combined_counter[(row['Street1'], row['Street2'])] += 1
        addresses['Street1_count'] = \
            addresses['Street1'].apply(lambda street1:
                                       self.street1_counter[street1])
        addresses['Street2_count'] = \
            addresses['Street2'].apply(lambda street2:
                                       self.street2_counter[street2])
        addresses['Street1_Street2_count'] = \
            [self.combined_counter[(row['Street1'], row['Street2'])]
             for i, row in addresses.iterrows()]
        addresses.drop((['Street1', 'Street2']
                        if self.params['include_Intersection'] else
                        ['Street1', 'Street2', 'Is_Intersection']),
                       axis=1, inplace=True)
        addresses.index = df.index
        df = pd.concat([df, addresses], axis=1)

        self.districts = df['PdDistrict'].unique()

        if self.params['include_Mean']:
            # Create column for mean priority of respective PdDistrict
            self.district_mean_priorities = \
                {district: y[df['PdDistrict'] == district].mean()
                 for district in self.districts}
            df['PdDistrict_Mean_Priority'] = \
                df['PdDistrict'].apply(lambda district:
                                       self.district_mean_priorities[district])

        # Fill missing latitudes and longitudes with mean of given PdDistrict
        missing_mask = df['Latitude'] == 90
        self.district_lat_lng_means = \
            {district:
             df[(~ missing_mask) &
                (df['PdDistrict'] == district)][['Longitude',
                                                 'Latitude']].mean()
             for district in self.districts}
        for i in df[missing_mask].index:
            df.loc[i, ['Longitude', 'Latitude']] = \
                self.district_lat_lng_means[df.loc[i, 'PdDistrict']]

        # Dummy PdDistrict
        if self.params['dummy_PdDistrict']:
            for district in self.districts[1:]:
                df['PdDistrict_%s' % district] = \
                    (df['PdDistrict'] == district).apply(int)
        if self.params['drop_PdDistrict']:
            df.drop('PdDistrict', axis=1, inplace=True)

        return df

    def fit(self, X, y):
        """Fits attributes to data.

        INPUT
        -----
        X : pandas.DataFrame
            Data from get_data, must include these columns:
            Dates, PdDistrict, Address, Latitude, Longitude, Priority
        y : numpy.array of shape [n_samples]
            Target values.

        OUTPUT
        ------
        self"""

        df = X.copy()

        # Convert Address column to Is_Intersection and three counter
        # columns for the two streets contained in the address and the
        # combination of the two
        addresses = pd.DataFrame(df['Address'].apply(split_address).tolist(),
                                 columns=['Is_Intersection', 'Street1',
                                          'Street2'])
        self.street1_counter = Counter()
        self.street2_counter = Counter()
        self.combined_counter = Counter()
        for i, row in addresses.iterrows():
            self.street1_counter[row['Street1']] += 1
            self.street2_counter[row['Street2']] += 1
            self.combined_counter[(row['Street1'], row['Street2'])] += 1

        self.districts = df['PdDistrict'].unique()

        # Create column for mean priority of respective PdDistrict
        if self.params['include_Mean']:
            self.district_mean_priorities = \
                {district: y[df['PdDistrict'] == district].mean()
                 for district in self.districts}

        # Fill missing latitudes and longitudes with mean of given PdDistrict
        missing_mask = df['Latitude'] == 90
        self.district_lat_lng_means = \
            {district:
             df[(~ missing_mask) &
                (df['PdDistrict'] == district)][['Longitude',
                                                 'Latitude']].mean()
             for district in self.districts}

        return self

    def transform(self, X):
        """Transform data.

        INPUT
        -----
        X : pandas.DataFrame
            Data from get_data, must include these columns:
            Dates, PdDistrict, Address, Latitude, Longitude, Priority

        OUTPUT
        ------
        df : pandas.DataFrame
            Data with feature engineering applied."""

        df = X.copy()

        # Convert Dates column to Year, Month, Day, Weekday (as int rather than
        # str), Hour (including minutes and seconds)
        df['Dates'] = df['Dates'].apply(lambda x:
                                        datetime.strptime(x,
                                                          '%Y-%m-%d %H:%M:%S'))
        df['DayOfYear'] = df['Dates'].apply(lambda x: x.dayofyear)
        df['DayOfWeek'] = df['Dates'].apply(lambda x: x.weekday())
        df['TimeOfDay'] = df['Dates'].apply(lambda x: x.hour + x.minute / 60. +
                                            x.second / 3600.)
        df.drop('Dates', axis=1, inplace=True)

        # Convert Address column to Is_Intersection and three counter
        # columns for the two streets contained in the address and the
        # combination of the two
        addresses = pd.DataFrame(df['Address'].apply(split_address).tolist(),
                                 columns=['Is_Intersection', 'Street1',
                                          'Street2'])
        df.drop('Address', axis=1, inplace=True)
        addresses['Street1_count'] = \
            addresses['Street1'].apply(lambda street1:
                                       self.street1_counter[street1])
        addresses['Street2_count'] = \
            addresses['Street2'].apply(lambda street2:
                                       self.street2_counter[street2])
        addresses['Street1_Street2_count'] = \
            [self.combined_counter[(row['Street1'], row['Street2'])]
             for i, row in addresses.iterrows()]
        addresses.drop((['Street1', 'Street2']
                        if self.params['include_Intersection'] else
                        ['Street1', 'Street2', 'Is_Intersection']),
                       axis=1, inplace=True)
        addresses.index = df.index
        df = pd.concat([df, addresses], axis=1)

        # Create column for mean priority of respective PdDistrict
        if self.params['include_Mean']:
            df['PdDistrict_Mean_Priority'] = \
                df['PdDistrict'].apply(lambda district:
                                       self.district_mean_priorities[district])

        # Fill missing latitudes and longitudes with mean of given PdDistrict
        missing_mask = df['Latitude'] == 90
        for i in df[missing_mask].index:
            df.loc[i, ['Longitude', 'Latitude']] = \
                self.district_lat_lng_means[df.loc[i, 'PdDistrict']]

        # Dummy PdDistrict
        if self.params['dummy_PdDistrict']:
            for district in self.districts[1:]:
                df['PdDistrict_%s' % district] = \
                    (df['PdDistrict'] == district).apply(int)
        if self.params['drop_PdDistrict']:
            df.drop('PdDistrict', axis=1, inplace=True)

        return df
