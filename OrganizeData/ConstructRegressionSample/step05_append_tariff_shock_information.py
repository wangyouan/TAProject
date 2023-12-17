#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: step05_append_tariff_shock_information
# @Date: 2023/12/17
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

"""
python -m OrganizeData.ConstructRegressionSample.step05_append_tariff_shock_information
"""

import os

import pandas as pd
from pandas import DataFrame

from Constants import Constants as const


def construct_post_event_dummy(tmp_df: DataFrame) -> DataFrame:
    tmp_df: DataFrame = tmp_df.sort_values(by=const.YEAR, ascending=True)
    end_year = tmp_df[const.YEAR].max()
    if tmp_df['tariff_cut_sic4'].sum() > 0:
        event_year = tmp_df.loc[tmp_df['tariff_cut_sic4'] == 1, const.YEAR].min()
        for year in range(event_year, end_year + 1):
            if abs(year - event_year) <= 3:
                tmp_df.loc[tmp_df[const.YEAR] == year, 'tariff_post3_sic4'] = 1
            if abs(year - event_year) <= 5:
                tmp_df.loc[tmp_df[const.YEAR] == year, 'tariff_post5_sic4'] = 1

            if year > event_year and tmp_df.loc[tmp_df[const.YEAR] == year, 'tariff_cut_sic4'].sum() == 1:
                event_year = year

    if tmp_df['tariff_cut_sic3'].sum() > 0:
        event_year = tmp_df.loc[tmp_df['tariff_cut_sic3'] == 1, const.YEAR].min()
        for year in range(event_year, end_year + 1):
            if abs(year - event_year) <= 3:
                tmp_df.loc[tmp_df[const.YEAR] == year, 'tariff_post3_sic3'] = 1
            if abs(year - event_year) <= 5:
                tmp_df.loc[tmp_df[const.YEAR] == year, 'tariff_post5_sic3'] = 1

            if year > event_year and tmp_df.loc[tmp_df[const.YEAR] == year, 'tariff_cut_sic3'].sum() == 1:
                event_year = year

    return tmp_df


if __name__ == '__main__':
    reg_df: DataFrame = pd.read_stata(
        os.path.join(const.RESULT_PATH, '20231110_pollution_regression_data_v2.dta')).dropna(
        subset=['ln1_air_harzard_1'])
    tariff_df_sic3: DataFrame = pd.read_csv(
        os.path.join(const.POSSIBLE_PROJECT_PATH, '20230816 Fluidity From GuMing', 'Tariff_cut_sic3.csv')).rename(
        columns={'tariff_cut': 'tariff_cut_sic3'})
    tariff_df_sic4: DataFrame = pd.read_csv(
        os.path.join(const.POSSIBLE_PROJECT_PATH, '20230816 Fluidity From GuMing', 'Tariff_cut_sic4.csv')).rename(
        columns={'tariff_cut': 'tariff_cut_sic4', 'sic4': 'sic'})
    reg_df.loc[:, 'sic_int'] = reg_df['sic'].astype(int)
    reg_df.loc[:, 'sic3_int'] = reg_df['sic3'].astype(int)
    reg_df = reg_df.drop([const.SIC_CODE, const.SIC3_CODE], axis=1).rename(
        columns={'sic_int': const.SIC_CODE, 'sic3_int': const.SIC3_CODE})

    reg_df2: DataFrame = reg_df.merge(tariff_df_sic3, on=[const.SIC3_CODE, const.YEAR], how='left').merge(
        tariff_df_sic4, on=[const.SIC_CODE, const.YEAR], how='left')

    for key in ['tariff_cut_sic4', 'tariff_cut_sic3']:
        reg_df2.loc[reg_df2[const.YEAR].apply(lambda x: 1990 <= x <= 2022), key] = reg_df2.loc[
            reg_df2[const.YEAR].apply(lambda x: 1990 <= x <= 2022), key].fillna(0)

    for key in ['tariff_post3_sic3', 'tariff_post5_sic3', 'tariff_post3_sic4', 'tariff_post5_sic4']:
        reg_df2.loc[:, key] = 0

    reg_df3: DataFrame = reg_df2.groupby(const.GVKEY).apply(construct_post_event_dummy).reset_index(drop=True)
    reg_df3.to_stata(os.path.join(const.RESULT_PATH, '20231217_pollution_regression_data.dta'), write_index=False)
