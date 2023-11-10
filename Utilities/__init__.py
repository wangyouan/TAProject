#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: __init__.py
# @Date: 2023/11/10
# @Author: Mark Wang
# @Email: wangyouan@gamil.com


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
