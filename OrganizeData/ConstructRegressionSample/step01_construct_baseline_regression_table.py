#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: step01_construct_baseline_regression_table
# @Date: 2023/9/21
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

"""
python -m OrganizeData.ConstructRegressionSample.step01_construct_baseline_regression_table
"""

import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from Constants import Constants as const

if __name__ == '__main__':
    ctat_df: DataFrame = pd.read_pickle(os.path.join(const.TEMP_PATH, '1950_2023_ctat_ta_ctrl_variables.pkl'))
    tax_effectiveness_df: DataFrame = pd.read_excel(os.path.join(
        const.DATABASE_PATH, 'TaxAvoidance', 'tax effectiveness', 'tax effectiveness full.xlsx'))
    same_sex_marriage_df: DataFrame = pd.read_csv(os.path.join(const.DATA_PATH, 'samesex_marriage_updated.csv')).loc[
                                      :, ['state_abbr', 'marriage']].rename(columns={'state_abbr': 'state'})
    same_sex_marriage_df.loc[:, 'marriage_date'] = pd.to_datetime(same_sex_marriage_df['marriage'], format='%Y/%m/%d')
    same_sex_marriage_df.loc[:, 'marriage_year'] = same_sex_marriage_df['marriage_date'].dt.year

    edgar_df: DataFrame = pd.read_excel(os.path.join(r'C:\Users\wyatc\OneDrive\Projects\PossibleProjects\EDGAR',
                                                     'EDGAR implementation.xlsx'))

    # shift tax avoidance measure
    ta_var_list = ['CASH_ETR', 'GAAP_ETR', 'CURRENT_ETR', 'TAPD_AT', 'DELTA_MVA', 'DELTA_BVA']
    for key in ta_var_list:
        for lag_year in range(1, 6):
            ctat_df.loc[:, '{}_{}'.format(key, lag_year)] = ctat_df.groupby(const.GVKEY)[key].shift(-lag_year)

    # merge tax effectiveness data
    ctat_df: DataFrame = ctat_df.merge(tax_effectiveness_df, on=[const.GVKEY, const.YEAR], how='left')
    for lag_year in range(1, 6):
        tmp_te_df: DataFrame = tax_effectiveness_df.copy()
        tmp_te_df.loc[:, const.YEAR] -= lag_year
        ctat_df: DataFrame = ctat_df.merge(tmp_te_df, on=[const.GVKEY, const.YEAR],
                                           suffixes=('', '_{}'.format(lag_year)), how='left')

    ctat_df.loc[:, const.SIC2_CODE] = ctat_df[const.SIC_CODE].str[:2]
    ctat_df.loc[:, const.SIC3_CODE] = ctat_df[const.SIC_CODE].str[:3]
    ctat_df_event: DataFrame = ctat_df.merge(same_sex_marriage_df, on=['state'], how='left').merge(
        edgar_df, on=[const.GVKEY], how='left')

    ctat_df_event.loc[:, 'after_marriage'] = (ctat_df_event['marriage_year'] < ctat_df_event['year']).astype(int)
    ctat_df_event.loc[:, 'post_marriage'] = (ctat_df_event['marriage_year'] <= ctat_df_event['year']).astype(int)
    ctat_df_event.loc[:, 'edg_year'] = ctat_df_event['edg_year'].fillna(1996)
    ctat_df_event.loc[:, 'after_edgar'] = (ctat_df_event['edg_year'] < ctat_df_event['year']).astype(int)
    ctat_df_event.loc[:, 'post_edgar'] = (ctat_df_event['edg_year'] <= ctat_df_event['year']).astype(int)
    ctat_df_event.loc[:, 'edgar_diff'] = ctat_df_event[const.YEAR] - ctat_df_event['edg_year']
    ctat_df_event.loc[:, 'marriage_diff'] = ctat_df_event[const.YEAR] - ctat_df_event['marriage_year']

    for year in range(1, 5):
        ctat_df_event.loc[:, 'y{}_edgar'.format(year)] = (ctat_df_event['edgar_diff'] == -year).astype(int)
        ctat_df_event.loc[:, 'edgar_y{}'.format(year)] = (ctat_df_event['edgar_diff'] == year).astype(int)
        ctat_df_event.loc[:, 'y{}_marriage'.format(year)] = (ctat_df_event['marriage_diff'] == -year).astype(int)
        ctat_df_event.loc[:, 'marriage_y{}'.format(year)] = (ctat_df_event['marriage_diff'] == year).astype(int)

    # keep only useful sample
    ctat_df_event2: DataFrame = ctat_df_event.loc[ctat_df_event[const.YEAR] >= 1988].copy()
    ctat_df_event2.loc[:, 'edgar_sample'] = ctat_df_event2[const.YEAR].apply(lambda x: int(1988 <= x <= 2001))
    ctat_df_event2.loc[:, 'marriage_sample'] = ctat_df_event2[const.YEAR].apply(lambda x: int(1999 <= x <= 2020))

    ctat_df_event2.to_stata(os.path.join(const.RESULT_PATH, '20230921_edgar_marriage_regression_data.dta'),
                            write_index=False)
