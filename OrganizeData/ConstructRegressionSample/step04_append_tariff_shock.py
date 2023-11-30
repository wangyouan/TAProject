#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: step04_append_tariff_shock
# @Date: 2023/11/17
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

"""
python -m OrganizeData.ConstructRegressionSample.step04_append_tariff_shock
"""

import os

import numpy as np
import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm

from Constants import Constants as const


def construct_tariff_event_id(sub_df):
    if sub_df['tariff_cut'].sum() == 0:
        return sub_df

    tariff_cut_year_list = list(sub_df.loc[sub_df['tariff_cut'] == 1, const.YEAR])
    sub_gvkey = sub_df.iloc[0][const.GVKEY]
    if len(tariff_cut_year_list) == 1:
        sub_df.loc[sub_df['tariff_cut'] == 1, 'tariff_event_id'] = '{}_{}'.format(sub_gvkey, tariff_cut_year_list[0])

    else:
        last_year = tariff_cut_year_list[0]
        sub_df.loc[sub_df[const.YEAR] == last_year, 'tariff_event_id'] = '{}_{}'.format(sub_gvkey, last_year)
        for next_year in tariff_cut_year_list[1:]:
            if next_year - last_year > 3:
                last_year = next_year
                sub_df.loc[sub_df[const.YEAR] == last_year, 'tariff_event_id'] = '{}_{}'.format(sub_gvkey, last_year)

    for event_id in sub_df['tariff_event_id'].dropna().drop_duplicates():
        event_year_j = int(event_id.split('_')[1])
        for j in sub_df.index:
            year_j = sub_df.loc[j, const.YEAR]
            diff_j = year_j - event_year_j
            if 0 < diff_j <= 3:
                sub_df.loc[j, 'tariff_event_a{}'.format(diff_j)] = 1
                sub_df.loc[j, 'tariff_post'.format(diff_j)] = 1
                sub_df.loc[j, 'tariff_event_id'] = event_id
            elif diff_j == 0:
                sub_df.loc[j, 'tariff_event'.format(diff_j)] = 1
                sub_df.loc[j, 'tariff_post'.format(diff_j)] = 1
            elif -3 <= diff_j < 0:
                sub_df.loc[j, 'tariff_event_b{}'.format(-diff_j)] = 1
                sub_df.loc[j, 'tariff_event_id'] = event_id

    return sub_df


if __name__ == '__main__':
    reg_df: DataFrame = pd.read_stata(
        os.path.join(const.RESULT_PATH, '20231110_pollution_regression_data_v2.dta')).dropna(
        subset=['ln1_air_harzard_1'])
    tariff_df: DataFrame = pd.read_csv(os.path.join(const.POSSIBLE_PROJECT_PATH, '20230816 Fluidity From GuMing',
                                                    'Tariff_cut_sic3.csv'))
    reg_df.loc[:, const.SIC3_CODE] = reg_df[const.SIC3_CODE].astype(int)
    tariff_df.loc[:, const.SIC3_CODE] = tariff_df[const.SIC3_CODE].astype(int)
    reg_df_tar: DataFrame = reg_df.merge(tariff_df, on=[const.SIC3_CODE, const.YEAR], how='left')
    reg_df_tar.loc[:, 'tariff_cut'] = reg_df_tar['tariff_cut'].fillna(0)
    reg_df_tar.loc[:, 'tariff_event'] = 0
    reg_df_tar.loc[:, 'tariff_event_id'] = np.nan

    reg_df_tar.loc[:, 'tariff_post'] = 0
    for lag_year in range(1, 4):
        reg_df_tar.loc[:, 'tariff_event_b{}'.format(lag_year)] = 0
        reg_df_tar.loc[:, 'tariff_event_a{}'.format(lag_year)] = 0
    reg_df_tar2: DataFrame = reg_df_tar.groupby(const.GVKEY).apply(construct_tariff_event_id).reset_index(drop=True)

    reg_df_tar2.loc[:, 'tariff_treated'] = reg_df_tar2.loc[:,
                                           ['tariff_event', 'tariff_event_b1', 'tariff_event_a1', 'tariff_event_b2',
                                            'tariff_event_a2', 'tariff_event_b3', 'tariff_event_a3']].sum(axis=1)
    reg_df_tar2.loc[:, 'tariff_treated'] = reg_df_tar2.loc[:, 'tariff_treated'].apply(lambda x: int(x > 0))
    reg_df_tar2.loc[:, 'psm_sample'] = reg_df_tar2.loc[:, 'tariff_treated']
    reg_df_tar2.loc[:, 'tariff_control'] = 0

    ctrl_vars = 'Size MktCap PROA Leverage'.split(' ')
    reg_df_tar3 = reg_df_tar2.dropna(subset=ctrl_vars, how='any').reset_index(drop=True)

    event_year_list = reg_df_tar3.loc[reg_df_tar3['tariff_event'] == 1, const.YEAR].drop_duplicates().sort_values()
    no_match_list = list()

    for every_event_year in event_year_list:
        match_df = reg_df_tar3.loc[reg_df_tar3[const.YEAR] == every_event_year].copy()
        to_match_list = match_df.loc[match_df['tariff_event'] == 0, const.GVKEY]
        match_gvkey_list = list(match_df.loc[match_df['tariff_event'] == 1, const.GVKEY])
        for target_gvkey in to_match_list:
            tmp_df = reg_df_tar3.loc[(reg_df_tar3[const.GVKEY] == target_gvkey)
                                     & reg_df_tar3[const.YEAR].apply(
                lambda x: every_event_year - 3 <= x <= every_event_year + 3)].copy()

            if tmp_df['psm_sample'].sum() == 0:
                match_gvkey_list.append(target_gvkey)

        psm_df = match_df.loc[match_df[const.GVKEY].isin(match_gvkey_list)].copy()
        X = psm_df[ctrl_vars]
        X = sm.add_constant(X)
        Y = psm_df['tariff_event']
        model = sm.OLS(Y, X)
        results = model.fit()
        psm_df.loc[:, 'predict_score'] = results.predict(X)
        to_match_gvkey_list = list(psm_df.loc[psm_df['tariff_event'] == 1, const.GVKEY])
        for gvkey in to_match_gvkey_list:
            matched_score = psm_df.loc[psm_df[const.GVKEY] == gvkey, 'predict_score'].iloc[0]

            # if there are not enough id to match
            if psm_df.loc[psm_df['psm_sample'] == 0].empty:
                no_match_list.append(gvkey)
                print(gvkey)
                continue

            match_id = (psm_df.loc[psm_df['psm_sample'] == 0, 'predict_score'] - matched_score).apply(abs).idxmin()
            psm_df.loc[match_id, 'psm_sample'] = 1
            matched_gvkey = psm_df.loc[match_id, 'gvkey']
            reg_df_tar3.loc[(reg_df_tar3[const.GVKEY] == matched_gvkey)
                            & reg_df_tar3[const.YEAR].apply(
                lambda x: every_event_year - 3 <= x <= every_event_year + 3), 'psm_sample'] = 1
            reg_df_tar3.loc[(reg_df_tar3[const.GVKEY] == matched_gvkey)
                            & reg_df_tar3[const.YEAR].apply(
                lambda x: every_event_year - 3 <= x <= every_event_year + 3), 'tariff_control'] = 1
            reg_df_tar3.loc[(reg_df_tar3[const.GVKEY] == matched_gvkey)
                            & reg_df_tar3[const.YEAR].apply(
                lambda x: every_event_year - 3 <= x <= every_event_year + 3), 'tariff_event_id'] = \
                reg_df_tar3.loc[(reg_df_tar3[const.GVKEY] == gvkey) & (
                        reg_df_tar3[const.YEAR] == every_event_year), 'tariff_event_id'].iloc[0]

    for gvkey in no_match_list:
        reg_df_tar3.loc[reg_df_tar3[const.GVKEY] == gvkey, 'psm_sample'] = -1

    reg_df_tar4: DataFrame = reg_df_tar3.copy()
    for gvkey in reg_df_tar4.loc[reg_df_tar4['tariff_control'] == 1, const.GVKEY].drop_duplicates():
        for i in reg_df_tar4.loc[(reg_df_tar4[const.GVKEY] == gvkey) & reg_df_tar4['tariff_event_id'].notnull()].index:
            event_year = int(reg_df_tar4.loc[i, 'tariff_event_id'].split('_')[-1])
            current_year = reg_df_tar4.loc[i, const.YEAR]
            diff = event_year - current_year
            if diff > 0:
                reg_df_tar4.loc[i, 'tariff_event_a{}'.format(diff)] = 1
                reg_df_tar4.loc[i, 'tariff_post'] = 1
            elif diff < 0:
                reg_df_tar4.loc[i, 'tariff_event_b{}'.format(-diff)] = 1
            else:
                reg_df_tar4.loc[i, 'tariff_event'] = 1
                reg_df_tar4.loc[i, 'tariff_post'] = 1

    reg_df_tar4[const.SIC3_CODE] = reg_df_tar4[const.SIC3_CODE].astype(int)
    reg_df_tar4.to_pickle(os.path.join(const.TEMP_PATH, '20231130_pollution_regression_data.pkl'))
    reg_df_tar4.to_stata(os.path.join(const.RESULT_PATH, '20231130_pollution_regression_data.dta'), write_index=False)
