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


def construct_event_post_dummy(sub_df):
    if sub_df['tariff_event'].sum() == 0:
        return sub_df

    event_year_df = sub_df.loc[sub_df['tariff_event'] == 1].copy()
    for i in sub_df.index:
        current_year = sub_df.loc[i, const.YEAR]
        diff = 100
        diff_abs = 100

        for event_year in event_year_df.year:
            tmp_diff = current_year - event_year
            if abs(tmp_diff) < diff_abs:
                diff_abs = abs(tmp_diff)
                diff = tmp_diff

        if 0 < diff_abs < 6:
            if diff > 0:
                sub_df.loc[i, 'tariff_event_p{}'.format(diff_abs)] = 1
            else:
                sub_df.loc[i, 'tariff_event_m{}'.format(diff_abs)] = 1

        if diff >= 0:
            sub_df.loc[i, 'tariff_post'] = 1

    return sub_df


if __name__ == '__main__':
    reg_df: DataFrame = pd.read_stata(os.path.join(const.RESULT_PATH, '20231110_pollution_regression_data_v2.dta'))
    tariff_df: DataFrame = pd.read_csv(os.path.join(const.POSSIBLE_PROJECT_PATH, '20230816 Fluidity From GuMing',
                                                    'Tariff_cut_sic3.csv'))
    reg_df.loc[:, const.SIC3_CODE] = reg_df[const.SIC3_CODE].astype(int)
    tariff_df.loc[:, const.SIC3_CODE] = tariff_df[const.SIC3_CODE].astype(int)
    reg_df_tar: DataFrame = reg_df.merge(tariff_df, on=[const.SIC3_CODE, const.YEAR], how='left')
    reg_df_tar.loc[:, 'tariff_cut'] = reg_df_tar['tariff_cut'].fillna(0)
    reg_df_tar.loc[:, 'tariff_event'] = (reg_df_tar['tariff_cut'] == 1).astype(int)
    reg_df_tar.loc[:, 'tariff_post'] = 0
    for lag_year in range(1, 6):
        reg_df_tar.loc[:, 'tariff_event_m{}'.format(lag_year)] = 0
        reg_df_tar.loc[:, 'tariff_event_p{}'.format(lag_year)] = 0

    reg_df_tar.loc[:, 'psm_sample'] = 0
    reg_df_tar.loc[:, 'psm_id'] = np.nan

    ctrl_vars = 'Size MktCap PROA Leverage'.split(' ')
    reg_df_tar3 = reg_df_tar.dropna(subset=ctrl_vars, how='any')
    event_gvkeys = reg_df_tar3.loc[reg_df_tar3['tariff_event'] == 1, 'gvkey'].drop_duplicates()
    no_match_list = list()
    for gvkey in event_gvkeys:
        event_df: DataFrame = reg_df_tar3.loc[reg_df_tar3[const.GVKEY] == gvkey].copy()
        match_year = event_df.loc[event_df['tariff_event'] == 1, const.YEAR].min()
        match_id = event_df.loc[event_df['tariff_event'] == 1, const.YEAR].idxmin()

        control_df: DataFrame = reg_df_tar3.loc[reg_df_tar3['psm_sample'] == 0].copy()
        control_df2: DataFrame = control_df.loc[control_df[const.YEAR] == match_year].copy()
        gvkey_list = control_df2[const.GVKEY].drop_duplicates()

        if gvkey_list.empty:
            print(gvkey, 'no potential gvkey list')
            no_match_list.append(gvkey)
            continue

        elif len(gvkey_list) == 1:
            matched_gvkey = gvkey_list[0]

        else:
            reg_df: DataFrame = control_df2.copy()
            reg_df2: DataFrame = pd.concat([reg_df, event_df.loc[[match_id]]], ignore_index=True)
            X = reg_df2[ctrl_vars]
            if X.shape[0] < 2:
                print(gvkey, 'no enough ctrl variables')
                no_match_list.append(gvkey)
                continue
            elif X.shape[0] == 2:
                matched_gvkey = reg_df['gvkey'].iloc[0]
            else:
                X = sm.add_constant(X)
                Y = reg_df2['tariff_event']
                if Y.sum() == 0:
                    print(gvkey, 'dependent all zero')
                    no_match_list.append(gvkey)
                    continue

                model = sm.OLS(Y, X)
                results = model.fit()

                X2 = reg_df[ctrl_vars]
                X2 = sm.add_constant(X2)
                reg_df.loc[:, 'predict_score'] = results.predict(X2)
                reg_df.loc[:, 'abs_diff'] = (reg_df.loc[:, 'predict_score'] - 1).apply(abs)
                matched_id = reg_df['abs_diff'].idxmin()
                matched_gvkey = reg_df.loc[matched_id, const.GVKEY]

        reg_df_tar3.loc[reg_df_tar3[const.GVKEY] == matched_gvkey, 'psm_sample'] = 1
        reg_df_tar3.loc[reg_df_tar3[const.GVKEY] == gvkey, 'psm_sample'] = 1
        reg_df_tar3.loc[reg_df_tar3[const.GVKEY] == matched_gvkey, 'psm_id'] = gvkey
        event_years = reg_df_tar3.loc[
            (reg_df_tar3[const.GVKEY] == gvkey) & (reg_df_tar3['tariff_event'] == 1), const.YEAR]
        for year in event_years:
            reg_df_tar3.loc[
                (reg_df_tar3[const.GVKEY] == matched_gvkey) & (reg_df_tar3['year'] == year), 'tariff_event'] = 1

        reg_df_tar3.loc[reg_df_tar3[const.GVKEY] == gvkey, 'psm_id'] = gvkey

    for gvkey in no_match_list:
        reg_df_tar3.loc[reg_df_tar3[const.GVKEY] == gvkey, 'psm_sample'] = -1

    reg_df_tar3: DataFrame = reg_df_tar3.groupby(const.GVKEY).apply(construct_event_post_dummy).reset_index(drop=True)
    reg_df_tar3[const.SIC3_CODE] = reg_df_tar3[const.SIC3_CODE].astype(int)
    reg_df_tar3.loc[:, 'tariff_treated'] = (reg_df_tar3['psm_id'] == reg_df_tar3[const.GVKEY]).astype(int)
    reg_df_tar3.to_pickle(os.path.join(const.TEMP_PATH, '20231127_pollution_regression_data.pkl'))
    reg_df_tar3.to_stata(os.path.join(const.RESULT_PATH, '20231127_pollution_regression_data.dta'), write_index=False)
