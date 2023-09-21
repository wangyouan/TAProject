#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: __init__.py
# @Date: 2023/9/20
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

from Constants.path_info import PathInfo


class Constants(PathInfo):
    # common identifier
    YEAR = 'year'
    CUSIP = 'cusip'
    TICKER = 'tic'
    CUSIP8 = 'cusip8'
    CUSIP6 = 'cusip6'
    GVKEY = 'gvkey'
    CIK = 'cik'

    # firm related variables
    SIC_CODE = 'sic'
    SIC2_CODE = 'sic2'
    SIC3_CODE = 'sic3'
    COMPANY_NAME = 'conm'
    EMPLOYEE_NUMBER = 'emp'
    STATE = 'state'
