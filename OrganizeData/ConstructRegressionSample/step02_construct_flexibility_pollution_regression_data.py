#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: step02_construct_flexibility_pollution_regression_data
# @Date: 2023/11/2
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

"""
python -m OrganizeData.ConstructRegressionSample.step02_construct_flexibility_pollution_regression_data
"""

import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from Constants import Constants as const

if __name__ == '__main__':
    fluidity_df: DataFrame = pd.read_csv(os.path.join(r'C:\Users\wyatc\OneDrive\Projects\PossibleProjects',
                                                      '20230816 From GuMing', 'FluidityData.csv'))
    pollution_df: DataFrame = pd.read_pickle(os.path.join(r'C:\Users\wyatc\OneDrive\Projects\GreenLoan\data\TRI',
                                                          'firm_level.pkl')).drop(
        'parent_company_name,parent_company_db_number,parent_company_name_standardized'.split(','), axis=1)
    ctat_df: DataFrame = pd.read_pickle(os.path.join(const.TEMP_PATH, '1950_2023_ctat_ta_ctrl_variables.pkl'))

    for key in ['air_release', 'water_release', 'land_release', 'total_release', 'sr_activities_num',
                'substitutions_num', 'product_num', 'process_modification_num', 'inventory_num', 'good_practics_num',
                'air_harzard', 'water_harzard', 'land_harzard', 'total_harzard', 'air_carcinogen', 'water_carcinogen',
                'land_carcinogen', 'total_carcinogen']:
        pollution_df.loc[:, 'ln1_{}'.format(key)] = pollution_df[key].apply(lambda x: np.log(x + 1))
        pollution_df.loc[:, 'ln_{}'.format(key)] = pollution_df[key].apply(np.log)

    reg_df: DataFrame = pollution_df.merge(fluidity_df, on=[const.GVKEY, const.YEAR], how='right').merge(ctat_df, on=[
        const.GVKEY, const.YEAR], how='inner')

    dep_keys = list(pollution_df.keys()[2:])
    all_dep_keys = dep_keys[:]
    for lag_year in range(1, 5):
        tmp_pollution_df: DataFrame = pollution_df.copy()
        tmp_pollution_df.loc[:, const.YEAR] -= lag_year
        reg_df: DataFrame = reg_df.merge(tmp_pollution_df, on=[const.GVKEY, const.YEAR], how='left',
                                         suffixes=('', '_{}'.format(lag_year)))
        all_dep_keys.extend(['{}_{}'.format(i, lag_year) for i in dep_keys])

    reg_df2: DataFrame = reg_df.replace([np.inf, -np.inf], np.nan).dropna(subset=all_dep_keys, how='all')
    reg_df2.to_stata(os.path.join(const.RESULT_PATH, '20231106_fluidity_pollution_regression_data.dta'),
                     write_index=False)
