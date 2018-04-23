#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:56:07 2018

@author: Das
"""

import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable

df = pd.read_csv('ex2data2.txt', sep = ',', header=None)
df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X = np.array(X)
Y = np.array(Y)

def linear_forward(A, W, b):
    z = torch.nn.linear()