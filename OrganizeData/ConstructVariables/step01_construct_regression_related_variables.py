#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: step01_construct_regression_related_variables
# @Date: 2023/9/20
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

"""
See notes about the list

python -m OrganizeData.ConstructVariables.step01_construct_regression_related_variables
"""

import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from Constants import Constants as const


def get_statutory_tax_rate(x):
    if 1942 <= x <= 1945:
        y = 0.4
    elif 1946 <= x <= 1949:
        y = 0.38
    elif x == 1950:
        y = 0.42
    elif x == 1951:
        y = 0.5075
    elif 1952 <= x <= 1963:
        y = 0.52
    elif x == 1964:
        y = 0.5
    elif 1965 <= x <= 1967:
        y = 0.48
    elif 1968 <= x <= 1969:
        y = 0.528
    elif x == 1970:
        y = 0.492
    elif 1971 <= x <= 1978:
        y = 0.48
    elif 1979 <= x <= 1986:
        y = 0.46
    elif x == 1987:
        y = 0.4
    elif 1988 <= x <= 1992:
        y = 0.34
    elif 1993 <= x <= 2017:
        y = 0.35
    # if x == 2018:
    else:
        y = 0.21
    return y


if __name__ == '__main__':
    ctat_df: DataFrame = pd.read_csv(os.path.join(const.DATABASE_PATH, 'compustat', '1950_2023_ctat_firm_ta_ctrl.zip'),
                                     dtype={const.CUSIP: str, const.SIC_CODE: str})
    ctat_df.loc[:, 'datadate'] = pd.to_datetime(ctat_df['datadate'], format='%Y-%m-%d')
    ctat_df.loc[:, const.YEAR] = ctat_df['fyear'].fillna(ctat_df['datadate'].apply(
        lambda x: x.year if x.month > 6 else x.year - 1))
    ctat_df.loc[:, 'statutory_tax_rate'] = ctat_df[const.YEAR].apply(get_statutory_tax_rate)
    ctat_df2: DataFrame = ctat_df.drop_duplicates(subset=[const.GVKEY, const.YEAR], keep='last').sort_values(
        by=[const.GVKEY, const.YEAR], ascending=True)
    us_ctat_df: DataFrame = ctat_df2.loc[ctat_df['fic'] == 'USA'].copy()
    us_ctat_df.loc[:, 'mkvalt'] = us_ctat_df['mkvalt'].fillna(us_ctat_df['prcc_f'] * us_ctat_df['csho'])
    us_ctat_df.loc[:, 'CASH_ETR'] = us_ctat_df['txpd'] / (us_ctat_df['pi'] - us_ctat_df['spi'])
    us_ctat_df.loc[:, 'GAAP_ETR'] = us_ctat_df['txt'] / (us_ctat_df['pi'] - us_ctat_df['spi'])
    us_ctat_df.loc[:, 'CURRENT_ETR'] = (us_ctat_df['txt'] - us_ctat_df['txdi']) / (us_ctat_df['pi'] - us_ctat_df['spi'])
    us_ctat_df.loc[:, 'lag_at'] = us_ctat_df.groupby(const.GVKEY)['at'].shift(1)
    us_ctat_df.loc[:, 'TAPD_AT'] = us_ctat_df['txpd'] / us_ctat_df['at']
    for key in ['CASH_ETR', 'GAAP_ETR', 'CURRENT_ETR']:
        us_ctat_df.loc[:, key] = us_ctat_df.apply(lambda x: x[key] if x['pi'] > 0 else np.nan, axis=1)
        us_ctat_df.loc[:, key] = us_ctat_df[key].apply(lambda x: 0 if x < 0 else 1 if x > 1 else x)

    winsorize_list = ['TAPD_AT']
