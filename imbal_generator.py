# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:37:36 2018

@author: Chiara
"""
import numpy as np

t = 8760
n = 15

# create noise for PV (only if PV nonzero) 
noise_PV_DA = np.random.normal(0,0.1,(t,n))

# create noise for Load
noise_Load_DA = np.random.normal(0,0.1,(t,n))


np.savetxt("noise_PV_DA.csv", noise_PV_DA , delimiter=",")
np.savetxt("noise_Load_DA.csv", noise_Load_DA , delimiter=",")



