# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:46:51 2017

@author: kamal
"""

import numpy as np

#labels = np.loadtxt("pca_a.txt", dtype="np.bytes", delimiter="\t", usecols=(4,), )

labels = np.loadtxt("pca_a.txt", dtype=bytes, delimiter="\t", usecols=(4,), )
print(labels)
print(labels.shape)

def decode_bytes(x):
    return x.decode('utf-8')

labels = [decode_bytes(x) for x in labels]
print(labels)
print(labels.size)

