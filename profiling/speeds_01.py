# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:17:19 2022

@author: rg5749
"""

from timer import Timer
method = 1
#***************************************
if method == 0:
    t = Timer()
    t.start()
    a = ()
    for i in range(10**5):
        a += (i,)
    t.stop()
    # 9.76 seconds
#***************************************
if method == 1:
    t = Timer()
    t.start()
    a = []
    for i in range(10**5):
        a.append(i)
    t.stop()
    # 0.0111 seconds
#***************************************
if method == 2:
    t = Timer()
    t.start()
    a = [i for i in range(10**5)]
    a = tuple(a)
    t.stop()
    # 0.0023 seconds