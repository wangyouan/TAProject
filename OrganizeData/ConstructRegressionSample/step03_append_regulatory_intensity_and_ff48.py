#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: step03_append_regulatory_intensity_and_ff48
# @Date: 2023/11/10
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

"""
python -m OrganizeData.ConstructRegressionSample.step03_append_regulatory_intensity_and_ff48
"""

import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from Constants import Constants as const

if __name__ == '__main__':
    ff48_df: DataFrame = pd.read_csv(os.path.join(const.DATABASE_PATH, 'fama french', 'ff_48ind.csv'),
                                     usecols=['sic', 'ff_48ind'], dtype={'sic': str})
    ff48_df.loc[:, 'sic'] = ff48_df[const.SIC_CODE].str.zfill(4)
    reg_df: DataFrame = pd.read_stata(
        os.path.join(const.RESULT_PATH, '20231110_fluidity_regulation_pollution_regression_data.dta'))

    reg_df2: DataFrame = reg_df.merge(ff48_df, on=[const.SIC_CODE], how='left')

    reg_intensity_df: DataFrame = pd.read_stata(
        os.path.join(r'C:\Users\wyatc\OneDrive\Projects\PossibleProjects', '20231109 Regulatory From GuMing',
                     'regulatory intensity files', 'regulatory intensity (year-firm).dta'))

    reg_df3: DataFrame = reg_df2.merge(reg_intensity_df, on=[const.GVKEY, const.YEAR], how='left')
    reg_df3.to_stata(os.path.join(const.RESULT_PATH, '20231110_pollution_regression_data_v2.dta'), write_index=False)
