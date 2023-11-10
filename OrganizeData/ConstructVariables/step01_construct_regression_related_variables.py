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
from scipy.stats.mstats import winsorize

from Constants import Constants as const
from Utilities import get_statutory_tax_rate

if __name__ == '__main__':
    ctat_df: DataFrame = pd.read_csv(os.path.join(const.DATABASE_PATH, 'compustat', '1950_2023_ctat_firm_ta_ctrl.zip'),
                                     dtype={const.CUSIP: str, const.SIC_CODE: str})
    ctat_df.loc[:, 'datadate'] = pd.to_datetime(ctat_df['datadate'], format='%Y-%m-%d')
    ctat_df.loc[:, const.YEAR] = ctat_df['fyear'].fillna(ctat_df['datadate'].apply(
        lambda x: x.year if x.month > 6 else x.year - 1))
    ctat_df.loc[:, 'statutory_tax_rate'] = ctat_df[const.YEAR].apply(get_statutory_tax_rate)
    ctat_df2: DataFrame = ctat_df.drop_duplicates(subset=[const.GVKEY, const.YEAR], keep='last').sort_values(
        by=[const.GVKEY, const.YEAR], ascending=True)

    # Filter out only USA data
    us_ctat_df: DataFrame = ctat_df2.loc[ctat_df['fic'] == 'USA'].copy()

    # Construct CASH_ETR GAAP_ETR CURRENT_ETR and TXPD_AT
    us_ctat_df.loc[:, 'mkvalt'] = us_ctat_df['mkvalt'].fillna(us_ctat_df['prcc_f'] * us_ctat_df['csho'])
    us_ctat_df.loc[:, 'CASH_ETR'] = us_ctat_df['txpd'] / (us_ctat_df['pi'] - us_ctat_df['spi'])
    us_ctat_df.loc[:, 'GAAP_ETR'] = us_ctat_df['txt'] / (us_ctat_df['pi'] - us_ctat_df['spi'])
    us_ctat_df.loc[:, 'CURRENT_ETR'] = (us_ctat_df['txt'] - us_ctat_df['txdi']) / (us_ctat_df['pi'] - us_ctat_df['spi'])
    us_ctat_df.loc[:, 'lag_at'] = us_ctat_df.groupby(const.GVKEY)['at'].shift(1)
    us_ctat_df.loc[:, 'TAPD_AT'] = us_ctat_df['txpd'] / us_ctat_df['at']

    # Truncate tax avoidance measure
    for key in ['CASH_ETR', 'GAAP_ETR', 'CURRENT_ETR']:
        us_ctat_df.loc[:, key] = us_ctat_df.apply(lambda x: x[key] if x['pi'] > 0 else np.nan, axis=1)
        us_ctat_df.loc[:, key] = us_ctat_df[key].apply(lambda x: 0 if x < 0 else 1 if x > 1 else x)

    # construct DELTA_MVA and DELTA_BVA
    us_ctat_df.loc[:, 'MVA'] = us_ctat_df['at'] + us_ctat_df['mkvalt'] - us_ctat_df['seq']
    us_ctat_df.loc[:, 'lag_MVA'] = us_ctat_df.groupby(const.GVKEY)['MVA'].shift(1)
    us_ctat_df.loc[:, 'lag_MVA'] = us_ctat_df.loc[:, 'lag_MVA'].fillna(us_ctat_df.loc[:, 'lag_at'])
    us_ctat_df.loc[:, 'lag_txr'] = us_ctat_df.groupby(const.GVKEY)['txr'].shift(1)
    us_ctat_df.loc[:, 'DELTA'] = (us_ctat_df['txpd'] - us_ctat_df['txr'] + us_ctat_df['lag_txr'] -
                                  us_ctat_df['statutory_tax_rate'] * us_ctat_df['pi'])
    us_ctat_df.loc[:, 'DELTA_MVA'] = us_ctat_df['DELTA'] / us_ctat_df['lag_MVA']
    us_ctat_df.loc[:, 'DELTA_BVA'] = us_ctat_df['DELTA'] / us_ctat_df['at']

    # Construct control variables
    us_ctat_df.loc[:, 'Size'] = us_ctat_df['at'].apply(np.log)
    us_ctat_df.loc[:, 'MktCap'] = us_ctat_df['mkvalt'].apply(np.log)
    us_ctat_df.loc[:, 'PROA'] = us_ctat_df['pi'] / us_ctat_df['lag_at']
    us_ctat_df.loc[:, 'Leverage'] = (us_ctat_df['dlc'] + us_ctat_df['dltt']) / us_ctat_df['lag_at']
    us_ctat_df.loc[:, 'NetPPE'] = us_ctat_df['ppent'] / us_ctat_df['lag_at']
    us_ctat_df.loc[:, 'RDRatio'] = us_ctat_df['xrd'].fillna(0) / us_ctat_df['lag_at']
    us_ctat_df.loc[:, 'SGA'] = us_ctat_df['xsga'].fillna(0) / us_ctat_df['lag_at']
    us_ctat_df.loc[:, 'NOL'] = us_ctat_df['tlcf'].fillna(0) / us_ctat_df['lag_at']
    us_ctat_df.loc[:, 'BTM'] = us_ctat_df['ceq'] / us_ctat_df['mkvalt']
    us_ctat_df.loc[:, 'Intangibles'] = us_ctat_df['intan'].fillna(0) / us_ctat_df['lag_at']
    us_ctat_df.loc[:, 'Foreign'] = us_ctat_df['pifo'].fillna(0) / us_ctat_df['lag_at']

    us_ctat_df2: DataFrame = us_ctat_df.replace([np.inf, -np.inf], np.nan)
    # Winsorize some variables
    winsorize_list = ['TAPD_AT', 'DELTA_MVA', 'DELTA_BVA', 'Size', 'MktCap', 'PROA', 'Leverage', 'NetPPE', 'RDRatio',
                      'SGA', 'NOL', 'BTM', 'Intangibles', 'Foreign']
    for key in winsorize_list:
        us_ctat_df.loc[us_ctat_df[key].notnull(), key] = winsorize(us_ctat_df[key].dropna(), limits=(0.01, 0.01))

    us_ctat_df2.to_pickle(os.path.join(const.TEMP_PATH, '1950_2023_ctat_ta_ctrl_variables.pkl'))
