# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:29:19 2018

@author: Chiara
"""

# SCENARIOS

import os
#import gurobipy as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shelve
import plotly.plotly as py
import plotly.tools as tls


dd1 = os.getcwd() #os.path.realpath(__file__) #os.getcwd()
data_path = str(Path(dd1).parent.parent)+r'\trunk\Input Data 2'
data_path2 = str(Path(dd1).parent.parent)+r'\branches\balancing'

filename=data_path+r'\input_data_2.out'
file_loc=data_path+r'\el_prices.xlsx'

from CT_DA import cost_DA, cost_DA_p, W_DA, W_DA_p
from CT_DApi import cost_DA_PI1_p, cost_DA_PI1, W_DA_PI1_p, W_DA_PI1
from TMST_def import TMST, TMST_run
#==============================================================================
# input data
#==============================================================================
d = shelve.open(filename, 'r')
el_price_dataframe = pd.read_excel(file_loc)/1000
el_price_notSampled = el_price_dataframe.values
el_price_sampled = el_price_notSampled[1::2]
el_price = el_price_sampled[:,1]

PV = d['PV']       
Load = d['Load']
Flex_load = d['Flex_load']

el_price_e = el_price

n = PV.shape[1]
tau = 0.1
window = 4

#==============================================================================
# create NOISE
#==============================================================================
# create noise for PV (only if PV nonzero) 
noise_PV_dataframe = pd.read_csv(data_path2+r'\noise_PV_DA.csv', header=None)
noise_PV = noise_PV_dataframe.values

for row in range(noise_PV.shape[0]):
    for col in range(noise_PV.shape[1]):
        if PV[row,col] == 0:
            noise_PV[row,col] = 0
            
# adding noise to PV (and if the result is less than zero bring it to zero)
PV_real = PV + noise_PV
for row in range(PV_real.shape[0]):
    for col in range(PV_real.shape[1]):
        if PV_real[row,col] < 0:
            PV_real[row,col] = 0

# create noise for Load
noise_Load_dataframe = pd.read_csv(data_path2+r'\noise_Load_DA.csv', header=None)
noise_Load = noise_Load_dataframe.values

# adding noise to Load (and if the result is less than zero bring it to zero)
Load_real = Load + noise_Load
for row in range(Load_real.shape[0]):
    for col in range(Load_real.shape[1]):
        if Load_real[row,col] < 0:
            Load_real[row,col] = 0

#==============================================================================
# # IMBALANCES
#==============================================================================
deltaPV = PV_real - PV
deltaLoad = Load_real - Load
deltaPV_prosumer = np.empty([n])
deltaLoad_prosumer = np.empty([n])
deltaPV_community = np.empty([TMST])
deltaLoad_community = np.empty([TMST])

for p in range(n):
    deltaPV_prosumer[p] = np.sum(deltaPV[:,p])
    deltaLoad_prosumer[p] = np.sum(deltaLoad[:,p])
for t in range(TMST):
    deltaPV_community[t] = np.sum(deltaPV[t,:])
    deltaLoad_community[t] = np.sum(deltaLoad[t,:])
    
imbalance_prosumer = deltaPV_prosumer - deltaLoad_prosumer
imbalance_community = deltaPV_community - deltaLoad_community
average_imbal_community = np.average(imbalance_community)

#==============================================================================
# PRICES
#==============================================================================

el_price_DA = el_price_e
el_price_DW = el_price_sampled[:,0]# 2 rows
el_price_UP = el_price_sampled[:,2]

system_state = np.empty([TMST])
for t in range(TMST):
    if el_price_DA[t] == el_price_DW[t]: #up-regulation
        system_state[t] = 1
    elif el_price_DA[t] == el_price_UP[t]: #dw-regulation
        system_state[t] = 2
    elif el_price_DW[t] == el_price_UP[t]: #balance
        system_state = 0

el_price_BAL = np.empty([TMST]) # 1 row 
for t in range(TMST):
    if system_state[t] == 0:
        el_price_BAL[t] = el_price_UP[t]
    elif system_state[t] == 1:
        el_price_BAL[t] = el_price_UP[t]
    elif system_state[t] == 2:
        el_price_BAL[t] = el_price_DW[t]

ret_price_exp = np.average(el_price_DW)
ret_price_imp = np.average(el_price_UP)

#%% SCENARIO 1

# imbalance costs
imbal_cost_tp_1 = np.zeros([TMST_run,n])
for t in range(TMST_run):
    for p in range(n):
        if (deltaPV[t,p]-deltaLoad[t,p]) < 0:
            imbal_cost_tp_1[t,p] = - ret_price_imp*(deltaPV[t,p]-deltaLoad[t,p])
        else:
            imbal_cost_tp_1[t,p] = - ret_price_exp*(deltaPV[t,p]-deltaLoad[t,p])
imbal_cost_p_1 = np.sum(imbal_cost_tp_1, axis = 0)
imbal_cost_1 = np.sum(imbal_cost_tp_1)
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_1_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_1_case1[p] = (cost_DA_p[p]+imbal_cost_p_1[p]-cost_DA_PI1_p[p])/abs(cost_DA_PI1_p[p])
costPercentage_1_case1 = (cost_DA + imbal_cost_1 - cost_DA_PI1)/abs(cost_DA_PI1)

WPercentage_p_1_case1 = np.zeros(n)
for p in range(n):
    WPercentage_p_1_case1[p] = (W_DA_p[p]+imbal_cost_p_1[p]-W_DA_PI1_p[p])/abs(W_DA_PI1_p[p])
WPercentage_1_case1 = (W_DA + imbal_cost_1 - W_DA_PI1)/abs(W_DA_PI1)

# QoE
numerator_1 = -imbal_cost_tp_1
denominator_1 = deltaPV - deltaLoad
perceived_price_1 = np.sum(numerator_1, axis=0)/np.sum(denominator_1, axis=0)
sigma_1 = np.std(perceived_price_1)
sigmaMax_1 = max(perceived_price_1) - min(perceived_price_1)
QoE_1 = 1 - sigma_1/sigmaMax_1

#%% SCENARIO 2 

# imbalance costs
imbal_cost_tp_2 = np.zeros([TMST_run,n])
for t in range(TMST_run):
    for p in range(n):
        if system_state[t] == 0:
            imbal_cost_tp_2[t,p] = - el_price_DA[t]*(deltaPV[t,p]-deltaLoad[t,p])
        if system_state[t] == 1:
            if (deltaPV[t,p]-deltaLoad[t,p]) < 0:
                imbal_cost_tp_2[t,p] = - el_price_UP[t]*(deltaPV[t,p]-deltaLoad[t,p])
            else:
                imbal_cost_tp_2[t,p] = -el_price_DA[t]*(deltaPV[t,p]-deltaLoad[t,p])
        if system_state[t] == 2:
            if (deltaPV[t,p]-deltaLoad[t,p]) < 0:
                imbal_cost_tp_2[t,p] = - el_price_DA[t]*(deltaPV[t,p]-deltaLoad[t,p])
            else:
                imbal_cost_tp_2[t,p] = -el_price_DW[t]*(deltaPV[t,p]-deltaLoad[t,p])
imbal_cost_p_2 = np.sum(imbal_cost_tp_2, axis = 0)
imbal_cost_2 = np.sum(imbal_cost_tp_2)
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_2_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_2_case1[p] = (cost_DA_p[p]+imbal_cost_p_2[p]-cost_DA_PI1_p[p])/abs(cost_DA_PI1_p[p])
costPercentage_2_case1 = (cost_DA + imbal_cost_2 - cost_DA_PI1)/abs(cost_DA_PI1)

WPercentage_p_2_case1 = np.zeros(n)
for p in range(n):
    WPercentage_p_2_case1[p] = (W_DA_p[p]+imbal_cost_p_2[p]-W_DA_PI1_p[p])/abs(W_DA_PI1_p[p])
WPercentage_2_case1 = (W_DA + imbal_cost_2 - W_DA_PI1)/abs(W_DA_PI1)
# QoE
numerator_2 = -imbal_cost_tp_2
denominator_2 = deltaPV - deltaLoad
perceived_price_2 = np.sum(numerator_2, axis=0)/np.sum(denominator_2, axis=0)
sigma_2 = np.std(perceived_price_2)
sigmaMax_2 = max(perceived_price_2) - min(perceived_price_2)
QoE_2 = 1 - sigma_2/sigmaMax_2

#%% SCENARIO 3

# imbalance costs
imbal_cost_t_3 = np.zeros([TMST_run])
for t in range(TMST_run):
    if imbalance_community[t] < 0:
        imbal_cost_t_3[t] = - (ret_price_imp)*(imbalance_community[t])
    else:
        imbal_cost_t_3[t] = - (ret_price_exp)*(imbalance_community[t])
imbal_cost_tp_3 = np.ones([TMST_run,n])
for t in range(TMST_run):
    for p in range(n):
        imbal_cost_tp_3[t,p] = imbal_cost_t_3[t]/n
imbal_cost_p_3 = np.sum(imbal_cost_tp_3, axis = 0)
imbal_cost_3 = np.sum(imbal_cost_t_3)
#imbal_weights = abs(imbalance_prosumer)/(abs(imbalance_prosumer).sum())
#for p in range(n):
#    imbal_cost_p_3[p] = imbal_weights[p]*imbal_cost_3
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_3_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_3_case1[p] = (cost_DA_p[p]+imbal_cost_p_3[p]-cost_DA_PI1_p[p])/abs(cost_DA_PI1_p[p])
costPercentage_3_case1 = (cost_DA + imbal_cost_3 - cost_DA_PI1)/abs(cost_DA_PI1)

WPercentage_p_3_case1 = np.zeros(n)
for p in range(n):
    WPercentage_p_3_case1[p] = (W_DA_p[p]+imbal_cost_p_3[p]-W_DA_PI1_p[p])/abs(W_DA_PI1_p[p])
WPercentage_3_case1 = (W_DA + imbal_cost_3 - W_DA_PI1)/abs(W_DA_PI1)

# QoE
numerator_3 = -imbal_cost_tp_3
denominator_3 = deltaPV - deltaLoad
perceived_price_3 = np.sum(numerator_3, axis=0)/np.sum(denominator_3, axis=0)
sigma_3 = np.std(perceived_price_3)
sigmaMax_3 = max(perceived_price_3) - min(perceived_price_3)
QoE_3 = 1 - sigma_3/sigmaMax_3

#%% SCENARIO 4

# imbalance costs
imbal_cost_t_4 = np.zeros([TMST_run])
for t in range(TMST_run):
    if system_state[t] == 0:
            imbal_cost_t_4[t,p] = - el_price_DA[t]*imbalance_community[t]
    if system_state[t] == 1:
        if imbalance_community[t] < 0:
            imbal_cost_t_4[t] = - el_price_UP[t]*imbalance_community[t]
        else:
            imbal_cost_t_4[t] = - el_price_DA[t]*imbalance_community[t]
    if system_state[t] == 2:
        if imbalance_community[t] < 0:
            imbal_cost_t_4[t] = - el_price_DA[t]*imbalance_community[t]
        else:
            imbal_cost_t_4[t] = - el_price_DW[t]*imbalance_community[t]
imbal_cost_tp_4 = np.ones([TMST_run,n])
for t in range(TMST_run):
    for p in range(n):
        imbal_cost_tp_4[t,p] = imbal_cost_t_4[t]/n
imbal_cost_p_4 = np.sum(imbal_cost_tp_4, axis = 0)
imbal_cost_4 = np.sum(imbal_cost_t_4)
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_4_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_4_case1[p] = (cost_DA_p[p]+imbal_cost_p_4[p]-cost_DA_PI1_p[p])/abs(cost_DA_PI1_p[p])
costPercentage_4_case1 = (cost_DA + imbal_cost_4 - cost_DA_PI1)/abs(cost_DA_PI1)

WPercentage_p_4_case1 = np.zeros(n)
for p in range(n):
    WPercentage_p_4_case1[p] = (W_DA_p[p]+imbal_cost_p_4[p]-W_DA_PI1_p[p])/abs(W_DA_PI1_p[p])
WPercentage_4_case1 = (W_DA + imbal_cost_4 - W_DA_PI1)/abs(W_DA_PI1)

# QoE
numerator_4 = -imbal_cost_tp_4
denominator_4 = deltaPV - deltaLoad
perceived_price_4 = np.sum(numerator_4, axis=0)/np.sum(denominator_4, axis=0)
sigma_4 = np.std(perceived_price_4)
sigmaMax_4 = max(perceived_price_4) - min(perceived_price_4)
QoE_4 = 1 - sigma_4/sigmaMax_4

#%% SCENARIO 5

from CT_bal5 import (CT_price2_sol_bal5, CT_q_sol_bal5, CT_beta_sol_bal5, 
CT_alfa_sol_bal5, CT_l_sol_bal5, CT_p_sol_bal5, mm_c_bal, mm_g_bal, y0_c_bal, y0_g_bal)
# imbalance costs
W_cost_tp_5 = np.empty([TMST_run,n])
imbal_cost_tp_5 = np.empty([TMST_run,n])
numerator_5 = np.empty([TMST_run,n])
denominator_5 = np.empty([TMST_run,n])
for t in range(TMST_run):
    for p in range(n):
        W_cost_tp_5[t,p] = (-CT_price2_sol_bal5[0,t])*(CT_q_sol_bal5[p,t]) - CT_beta_sol_bal5[p,t]*ret_price_exp + CT_alfa_sol_bal5[p,t]*(ret_price_imp) + \
        y0_c_bal[t,p]*CT_l_sol_bal5[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal5[p,t]*CT_l_sol_bal5[p,t] + y0_g_bal[t,p]*CT_p_sol_bal5[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal5[p,t]*CT_p_sol_bal5[p,t]
for t in range(TMST_run):
    for p in range(n):
        imbal_cost_tp_5[t,p] = (-CT_price2_sol_bal5[0,t])*(CT_q_sol_bal5[p,t])- CT_beta_sol_bal5[p,t]*ret_price_exp + CT_alfa_sol_bal5[p,t]*(ret_price_imp) 
        numerator_5[t,p] = -(-CT_price2_sol_bal5[0,t])*(CT_q_sol_bal5[p,t]) + CT_beta_sol_bal5[p,t]*ret_price_exp - CT_alfa_sol_bal5[p,t]*ret_price_imp
        denominator_5[t,p] = CT_beta_sol_bal5[p,t] - CT_alfa_sol_bal5[p,t] - CT_q_sol_bal5[p,t]

imbal_cost_p_5 = np.sum(imbal_cost_tp_5, axis=0)
imbal_cost_5 = np.sum(imbal_cost_p_5)

W_cost_p_5 = np.sum(W_cost_tp_5, axis=0)
W_cost_5 = np.sum(W_cost_p_5)

# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_5_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_5_case1[p] = (cost_DA_p[p]+imbal_cost_p_5[p]-cost_DA_PI1_p[p])/abs(cost_DA_PI1_p[p])
costPercentage_5_case1 = (cost_DA + imbal_cost_5 - cost_DA_PI1)/abs(cost_DA_PI1)

WPercentage_p_5_case1 = np.zeros(n)
for p in range(n):
    WPercentage_p_5_case1[p] = (W_DA_p[p]+W_cost_p_5[p]-W_DA_PI1_p[p])/abs(W_DA_PI1_p[p])
WPercentage_5_case1 = (W_DA + W_cost_5 - W_DA_PI1)/abs(W_DA_PI1)

# QoE
perceived_price_5 = np.sum(numerator_5, axis = 0)/np.sum(denominator_5, axis = 0)
sigma_5 = np.std(perceived_price_5)
sigmaMax_5 = max(perceived_price_5) - min(perceived_price_5)
QoE_5 = 1 - sigma_5/sigmaMax_5

#%% SCENARIO 6
from CT_bal6 import (CT_price2_sol_bal6, CT_q_sol_bal6, CT_beta_sol_bal6, 
CT_alfa_sol_bal6, CT_l_sol_bal6, CT_p_sol_bal6, mm_c_bal, mm_g_bal, y0_c_bal, y0_g_bal)
# imbalance costs
imbal_cost_tp_6 = np.empty([TMST_run,n])
numerator_6 = np.empty([TMST_run,n])
denominator_6 = np.empty([TMST_run,n])
W_cost_tp_6 = np.empty([TMST_run,n])
for t in range(TMST_run):
    for p in range(n):
        if system_state[t] == 2:
            W_cost_tp_6[t,p] = (-CT_price2_sol_bal6[0,t])*(CT_q_sol_bal6[p,t]) - CT_beta_sol_bal6[p,t]*el_price_DW[t] + CT_alfa_sol_bal6[p,t]*el_price_DA[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal6[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal6[p,t]*CT_l_sol_bal6[p,t] + y0_g_bal[t,p]*CT_p_sol_bal6[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal6[p,t]*CT_p_sol_bal6[p,t]
        elif system_state[t] == 1:
            W_cost_tp_6[t,p] = (-CT_price2_sol_bal6[0,t])*(CT_q_sol_bal6[p,t]) - CT_beta_sol_bal6[p,t]*el_price_DA[t] + CT_alfa_sol_bal6[p,t]*el_price_UP[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal6[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal6[p,t]*CT_l_sol_bal6[p,t] + y0_g_bal[t,p]*CT_p_sol_bal6[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal6[p,t]*CT_p_sol_bal6[p,t]
        else:
            W_cost_tp_6[t,p] = (-CT_price2_sol_bal6[0,t])*(CT_q_sol_bal6[p,t]) - CT_beta_sol_bal6[p,t]*el_price_DA[t] + CT_alfa_sol_bal6[p,t]*el_price_DA[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal6[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal6[p,t]*CT_l_sol_bal6[p,t] + y0_g_bal[t,p]*CT_p_sol_bal6[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal6[p,t]*CT_p_sol_bal6[p,t]
for t in range(TMST_run):
    for p in range(n):
        if system_state[t] == 2:
            imbal_cost_tp_6[t,p] = (-CT_price2_sol_bal6[0,t])*(CT_q_sol_bal6[p,t])- CT_beta_sol_bal6[p,t]*el_price_DW[t] + CT_alfa_sol_bal6[p,t]*el_price_DA[t] 
        elif system_state[t] == 1:
            imbal_cost_tp_6[t,p] = (-CT_price2_sol_bal6[0,t])*(CT_q_sol_bal6[p,t])- CT_beta_sol_bal6[p,t]*el_price_DA[t] + CT_alfa_sol_bal6[p,t]*el_price_UP[t]
        else:    
            imbal_cost_tp_6[t,p] = (-CT_price2_sol_bal6[0,t])*(CT_q_sol_bal6[p,t])- CT_beta_sol_bal6[p,t]*el_price_DA[t] + CT_alfa_sol_bal6[p,t]*el_price_DA[t]
        numerator_6[t,p] = -(-CT_price2_sol_bal6[0,t])*(CT_q_sol_bal6[p,t]) + CT_beta_sol_bal6[p,t]*el_price_DW[t] - CT_alfa_sol_bal6[p,t]*el_price_UP[t]
        denominator_6[t,p] = CT_beta_sol_bal6[p,t] - CT_alfa_sol_bal6[p,t] - CT_q_sol_bal6[p,t]

imbal_cost_p_6= np.sum(imbal_cost_tp_6, axis=0)
imbal_cost_6= np.sum(imbal_cost_p_6) 

W_cost_p_6= np.sum(W_cost_tp_6, axis=0)
W_cost_6= np.sum(W_cost_p_6) 

costPercentage_p_6_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_6_case1[p] = (cost_DA_p[p]+imbal_cost_p_6[p]-cost_DA_PI1_p[p])/abs(cost_DA_PI1_p[p])
costPercentage_6_case1 = (cost_DA + imbal_cost_6 - cost_DA_PI1)/abs(cost_DA_PI1)

WPercentage_p_6_case1 = np.zeros(n)
for p in range(n):
    WPercentage_p_6_case1[p] = (W_DA_p[p]+W_cost_p_6[p]-W_DA_PI1_p[p])/abs(W_DA_PI1_p[p])
WPercentage_6_case1 = (W_DA + W_cost_6 - W_DA_PI1)/abs(W_DA_PI1)

# QoE
perceived_price_6 = np.sum(numerator_6, axis = 0)/np.sum(denominator_6, axis = 0)
sigma_6 = np.std(perceived_price_6)
sigmaMax_6 = max(perceived_price_6) - min(perceived_price_6)
QoE_6 = 1 - sigma_6/sigmaMax_6

#%% SCENARIO 6 - resP
from CT_bal6resP import (CT_price2_sol_bal_resP, CT_q_sol_bal_resP, CT_beta_sol_bal_resP, 
CT_alfa_sol_bal_resP, CT_l_sol_bal_resP, CT_p_sol_bal_resP, mm_c_bal_resP, mm_g_bal_resP, y0_c_bal_resP, y0_g_bal_resP)
from CT_resP import cost_resP_p, cost_resP, CT_r_p_UP_sol_resP, CT_r_l_UP_sol_resP, W_resP, W_resP_p
# imbalance costs
imbal_cost_tp_6resP = np.empty([TMST_run,n])
numerator_6resP = np.empty([TMST_run,n])
denominator_6resP = np.empty([TMST_run,n])
W_cost_tp_6resP = np.empty([TMST_run,n])
for t in range(TMST_run):
    for p in range(n):
        if system_state[t] == 2:
            W_cost_tp_6resP[t,p] = (-CT_price2_sol_bal_resP[0,t])*(CT_q_sol_bal_resP[p,t]) - CT_beta_sol_bal_resP[p,t]*el_price_DW[t] + CT_alfa_sol_bal_resP[p,t]*el_price_DA[t] + \
            y0_c_bal_resP[t,p]*CT_l_sol_bal_resP[p,t] + mm_c_bal_resP[t,p]/2*CT_l_sol_bal_resP[p,t]*CT_l_sol_bal_resP[p,t] + y0_g_bal_resP[t,p]*CT_p_sol_bal_resP[p,t] + mm_g_bal_resP[t,p]/2*CT_p_sol_bal_resP[p,t]*CT_p_sol_bal_resP[p,t]
        elif system_state[t] == 1:
            W_cost_tp_6resP[t,p] = (-CT_price2_sol_bal_resP[0,t])*(CT_q_sol_bal_resP[p,t]) - CT_beta_sol_bal_resP[p,t]*el_price_DA[t] + CT_alfa_sol_bal_resP[p,t]*el_price_UP[t] + \
            y0_c_bal_resP[t,p]*CT_l_sol_bal_resP[p,t] + mm_c_bal_resP[t,p]/2*CT_l_sol_bal_resP[p,t]*CT_l_sol_bal_resP[p,t] + y0_g_bal_resP[t,p]*CT_p_sol_bal_resP[p,t] + mm_g_bal_resP[t,p]/2*CT_p_sol_bal_resP[p,t]*CT_p_sol_bal_resP[p,t]
        else:
            W_cost_tp_6resP[t,p] = (-CT_price2_sol_bal_resP[0,t])*(CT_q_sol_bal_resP[p,t]) - CT_beta_sol_bal_resP[p,t]*el_price_DA[t] + CT_alfa_sol_bal_resP[p,t]*el_price_DA[t] + \
            y0_c_bal_resP[t,p]*CT_l_sol_bal_resP[p,t] + mm_c_bal_resP[t,p]/2*CT_l_sol_bal_resP[p,t]*CT_l_sol_bal_resP[p,t] + y0_g_bal_resP[t,p]*CT_p_sol_bal_resP[p,t] + mm_g_bal_resP[t,p]/2*CT_p_sol_bal_resP[p,t]*CT_p_sol_bal_resP[p,t]
for t in range(TMST_run):
    for p in range(n):
        if system_state[t] == 2:
            imbal_cost_tp_6resP[t,p] =(-CT_price2_sol_bal_resP[0,t])*(CT_q_sol_bal_resP[p,t])- CT_beta_sol_bal_resP[p,t]*el_price_DW[t] + CT_alfa_sol_bal_resP[p,t]*el_price_DA[t] 
        elif system_state[t] == 1:
            imbal_cost_tp_6resP[t,p] =(-CT_price2_sol_bal_resP[0,t])*(CT_q_sol_bal_resP[p,t])- CT_beta_sol_bal_resP[p,t]*el_price_DA[t] + CT_alfa_sol_bal_resP[p,t]*el_price_UP[t] 
        else:
            imbal_cost_tp_6resP[t,p] =(-CT_price2_sol_bal_resP[0,t])*(CT_q_sol_bal_resP[p,t]) - CT_beta_sol_bal_resP[p,t]*el_price_DA[t] + CT_alfa_sol_bal_resP[p,t]*el_price_DA[t] 
        numerator_6resP[t,p] = -(-CT_price2_sol_bal_resP[0,t])*(CT_q_sol_bal_resP[p,t]) + CT_beta_sol_bal_resP[p,t]*el_price_DW[t] - CT_alfa_sol_bal_resP[p,t]*el_price_UP[t]
        denominator_6resP[t,p] = CT_beta_sol_bal_resP[p,t] - CT_alfa_sol_bal_resP[p,t] - CT_q_sol_bal_resP[p,t]

imbal_cost_p_6resP= np.sum(imbal_cost_tp_6resP, axis=0)
imbal_cost_6resP= np.sum(imbal_cost_p_6resP)

W_cost_p_6resP= np.sum(W_cost_tp_6resP, axis=0)
W_cost_6resP= np.sum(W_cost_p_6resP)

# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_6resP_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_6resP_case1[p] = (cost_resP_p[p]+imbal_cost_p_6resP[p]-cost_DA_PI1_p[p])/abs(cost_DA_PI1_p[p])
costPercentage_6resP_case1 = (cost_resP + imbal_cost_6resP - cost_DA_PI1)/abs(cost_DA_PI1)

WPercentage_p_6resP_case1 = np.zeros(n)
for p in range(n):
    WPercentage_p_6resP_case1[p] = (W_resP_p[p] + W_cost_p_6resP[p]-W_DA_PI1_p[p])/abs(W_DA_PI1_p[p])
WPercentage_6resP_case1 = (W_resP + W_cost_6resP - W_DA_PI1)/abs(W_DA_PI1)

# QoE
perceived_price_6resP = np.sum(numerator_6resP, axis = 0)/np.sum(denominator_6resP, axis = 0)
sigma_6resP = np.std(perceived_price_6resP)
sigmaMax_6resP = max(perceived_price_6resP) - min(perceived_price_6resP)
QoE_6resP = 1 - sigma_6resP/sigmaMax_6resP
res_UP_distribution = np.zeros([TMST_run,n])
for t in range(TMST_run):
    for p in range(n):
        res_UP_distribution[t,p] = CT_r_p_UP_sol_resP[p,t] + CT_r_l_UP_sol_resP[p,t]
res_UP_distribution_p = np.sum(res_UP_distribution, axis=0)


#%% save results in DataFrame

percentage_costs_p = pd.DataFrame(np.array([costPercentage_p_1_case1,costPercentage_p_2_case1,costPercentage_p_3_case1,costPercentage_p_4_case1,costPercentage_p_5_case1,costPercentage_p_6_case1,costPercentage_p_6resP_case1]).T, columns=['costPercentage s1','costPercentage s2','costPercentage s3','costPercentage s4','costPercentage s5','costPercentage s6','costPercentage s6resP'])
percentage_costs = pd.DataFrame(np.array([costPercentage_1_case1,costPercentage_2_case1,costPercentage_3_case1,costPercentage_4_case1,costPercentage_5_case1,costPercentage_6_case1,costPercentage_6resP_case1]).reshape(1,-1), columns=['costPercentage s1','costPercentage s2','costPercentage s3','costPercentage s4','costPercentage s5','costPercentage s6','costPercentage s6resP'])

W_p = pd.DataFrame(np.array([WPercentage_p_1_case1,WPercentage_p_2_case1,WPercentage_p_3_case1,WPercentage_p_4_case1,WPercentage_p_5_case1,WPercentage_p_6_case1,WPercentage_p_6resP_case1]).T, columns=['costPercentage s1','costPercentage s2','costPercentage s3','costPercentage s4','costPercentage s5','costPercentage s6','costPercentage s6resP'])
W = pd.DataFrame(np.array([WPercentage_1_case1,WPercentage_2_case1,WPercentage_3_case1,WPercentage_4_case1,WPercentage_5_case1,WPercentage_6_case1,WPercentage_6resP_case1]).reshape(1,-1), columns=['costPercentage s1','costPercentage s2','costPercentage s3','costPercentage s4','costPercentage s5','costPercentage s6','costPercentage s6resP'])

imbal_costs = pd.DataFrame(np.array([imbal_cost_1,imbal_cost_2,imbal_cost_3,imbal_cost_4,imbal_cost_5,imbal_cost_6,imbal_cost_6resP]).reshape(1,-1), columns=['imbal cost s1','imbal cost s2','imbal cost s3','imbal cost s4','imbal cost s5','imbal cost s6','imbal cost s6resP'])
imbal_costs_p = pd.DataFrame(np.array([imbal_cost_p_1,imbal_cost_p_2,imbal_cost_p_3,imbal_cost_p_4,imbal_cost_p_5,imbal_cost_p_6,imbal_cost_p_6resP]).T, columns=['imbal cost s1','imbal cost s2','imbal cost s3','imbal cost s4','imbal cost s5','imbal cost s6','imbal cost s6resP'])

twoStage_costs = pd.DataFrame(np.array([cost_DA+imbal_cost_1,cost_DA+imbal_cost_2,cost_DA+imbal_cost_3,cost_DA+imbal_cost_4,cost_DA+imbal_cost_5,cost_DA+imbal_cost_6,cost_resP+imbal_cost_6resP]).reshape(1,-1), columns=['two stages cost s1','two stages cost s2','two stages cost s3','two stages cost s4','two stages cost s5','two stages cost s6','two stages cost s6resP'])

QoE = pd.DataFrame(np.array([QoE_1,QoE_2,QoE_3,QoE_4,QoE_5,QoE_6,QoE_6resP]).reshape(1,-1), columns=['imbal cost s1','imbal cost s2','imbal cost s3','imbal cost s4','imbal cost s5','imbal cost s6','imbal cost s6resP'])

#prova_00_6 = cost_DA_tp[0,0] + imbal_cost_tp_6[0,0]
#prova_00_6resP = cost_resP_tp[0,0] + imbal_cost_tp_6resP[0,0]
#prova_00 = (cost_DA_tp + imbal_cost_tp_6>cost_resP_tp + imbal_cost_tp_6resP)

#%% pictures

# imbalance costs per prosumer accross scenarios
fig_imbalCost_p = plt.figure(1,figsize=[10,6])
ax1, = plt.plot(imbal_cost_p_1,label='scenario 1F')
ax2, = plt.plot(imbal_cost_p_2,label='scenario 1D')
ax3, = plt.plot(imbal_cost_p_3,label='scenario 2F')
ax4, = plt.plot(imbal_cost_p_4,label='scenario 2D')
ax5, = plt.plot(imbal_cost_p_5,label='scenario 3F',linestyle='dashed')
ax6, = plt.plot(imbal_cost_p_6,label='scenario 3D',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4, ax5, ax6], ['scenario 1F', 'scenario 1D','scenario 2F', 'scenario 2D','scenario 3F', 'scenario 3D'])
plt.xlabel('prosumer index')
plt.ylabel('balancing cost[AUD/kWh]')
plt.title('balancing cost per prosumer across scenarios')
plt.savefig('figures/imbalCost_p_scenarios.png', bbox_inches='tight')

# revenues from balancing stage per prosumer accross scenarios - PLOOOT appendix
rev_pros_6 = np.array(- imbal_cost_p_6)
fig_imbalCost_p = plt.figure(1,figsize=[10,6])
ax1, = plt.plot(- imbal_cost_p_1,label='scenario 1F')
ax2, = plt.plot(- imbal_cost_p_2,label='scenario 1D')
ax3, = plt.plot(- imbal_cost_p_3,label='scenario 2F')
ax4, = plt.plot(- imbal_cost_p_4,label='scenario 2D')
ax5, = plt.plot(- imbal_cost_p_5,label='scenario 3F',linestyle='dashed')
ax6, = plt.plot(- imbal_cost_p_6,label='scenario 3D',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4, ax5, ax6], ['scenario 1F', 'scenario 1D','scenario 2F', 'scenario 2D','scenario 3F', 'scenario 3D'])
plt.xlabel('Prosumer index')
plt.ylabel('Balancing stage revenues over the whole year [AUD]')
#plt.title('balancing cost per prosumer across scenarios')
plt.savefig('figures/rAbs_p_s1-6.png', bbox_inches='tight')

# welfare from balancing stage per prosumer accross scenarios - PLOOOT appendix
welf_pros_6 = np.array( - W_cost_p_6)
fig_imbalCost_p = plt.figure(2,figsize=[10,6])
ax1, = plt.plot(- imbal_cost_p_1,label='scenario 1F')
ax2, = plt.plot(- imbal_cost_p_2,label='scenario 1D')
ax3, = plt.plot(- imbal_cost_p_3,label='scenario 2F')
ax4, = plt.plot(- imbal_cost_p_4,label='scenario 2D')
ax5, = plt.plot(- W_cost_p_5,label='scenario 3F',linestyle='dashed')
ax6, = plt.plot(- W_cost_p_6,label='scenario 3D',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4, ax5, ax6], ['scenario 1F', 'scenario 1D','scenario 2F', 'scenario 2D','scenario 3F', 'scenario 3D'])
plt.xlabel('Prosumer index')
plt.ylabel('Balancing stage welfare over the whole year [AUD]')
#plt.title('balancing cost per prosumer across scenarios')
plt.savefig('figures/wAbs_p_s1-6.png', bbox_inches='tight')

# TRY -- PLOT!!
n_groups = 15
means_frank = rev_pros_6
means_guido = welf_pros_6
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, means_frank, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Revenues')
rects2 = plt.bar(index + bar_width, means_guido, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Welfare') 
plt.xlabel('Prosumer index')
plt.ylabel('Balancing stage welfare/revenues over the whole year [AUD]')
#plt.title('Scores by person')
plt.xticks(index + bar_width, ('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'))
plt.legend()
plt.tight_layout()
plt.savefig('figures/revWelf_6.png', bbox_inches='tight')

# percentage deviation costs per prosumer accross scenarios
fig_percCost_p = plt.figure(2,figsize=[10,6])
ax1, = plt.plot(-WPercentage_p_1_case1,label='scenario 1F')
ax2, = plt.plot(-WPercentage_p_2_case1,label='scenario 1D')
ax3, = plt.plot(-WPercentage_p_3_case1,label='scenario 2F')
ax4, = plt.plot(-WPercentage_p_4_case1,label='scenario 2D')
ax5, = plt.plot(-WPercentage_p_5_case1,label='scenario 3F',linestyle='dashed')
ax6, = plt.plot(-WPercentage_p_6_case1,label='scenario 3D',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4, ax5, ax6], ['scenario 1F', 'scenario 1D','scenario 2F', 'scenario 2D','scenario 3F', 'scenario 3D'])
plt.xlabel('prosumer index')
plt.ylabel('deviation costs [%]')
plt.title('Percentage deviation costs from perfect information per prosumer across scenarios')
plt.savefig('figures/wAbs_p_s1-6.png', bbox_inches='tight')

# percentage deviation costs per prosumer accross scenarios
fig_percCost_p = plt.figure(3,figsize=[10,6])
ax1, = plt.plot(costPercentage_p_1_case1,label='scenario 1F')
ax2, = plt.plot(costPercentage_p_2_case1,label='scenario 1D')
ax3, = plt.plot(costPercentage_p_3_case1,label='scenario 2F')
ax4, = plt.plot(costPercentage_p_4_case1,label='scenario 2D')
ax5, = plt.plot(costPercentage_p_5_case1,label='scenario 3F',linestyle='dashed')
ax6, = plt.plot(costPercentage_p_6_case1,label='scenario 3D',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4, ax5, ax6], ['scenario 1F', 'scenario 1D','scenario 2F', 'scenario 2D','scenario 3F', 'scenario 3D'])
plt.xlabel('prosumer index')
plt.ylabel('deviation costs [%]')
plt.title('Percentage deviation costs from perfect information per prosumer across scenarios')
plt.savefig('figures/percCost_p_scenarios.png', bbox_inches='tight')

# total imbalance costs accross scenarios -- PLOT
imbal_cost_135 = np.array((-imbal_cost_1,-imbal_cost_3,-W_cost_5))
imbal_cost_246 = np.array((-imbal_cost_2,-imbal_cost_4,-W_cost_6))
xlab = ['Scenario 1','Scenario 2','Scenario 3']
tot_cost = plt.figure(4,figsize=[10,6])
ax = tot_cost.add_subplot(111)
res = pd.DataFrame([imbal_cost_135, imbal_cost_246],index=['Fixed tariff', 'Dynamic tariff'],columns=xlab).transpose()
df_plot = pd.DataFrame([imbal_cost_135, imbal_cost_246],index=['Fixed tariff', 'Dynamic tariff'],columns=xlab).transpose()
df_plot.plot(kind='bar',ax=ax)
plt.xticks(rotation=360)
plt.ylabel('Total imbalance costs [AUD]')
plt.tight_layout()
plt.grid()
plt.savefig('figures/wAbs_s1-6.png', bbox_inches='tight')

# percentage costs accross scenarios
perc_cost_135 = np.array((costPercentage_1_case1,costPercentage_3_case1,costPercentage_5_case1))
perc_cost_246 = np.array((costPercentage_2_case1,costPercentage_4_case1,costPercentage_6_case1))
xlab = ['Scenario 1','Scenario 2','Scenario 3']
tot_cost = plt.figure(5,figsize=[10,6])
ax = tot_cost.add_subplot(111)
res = pd.DataFrame([perc_cost_135, perc_cost_246],index=['Fixed tariff', 'Dynamic tariff'],columns=xlab).transpose()
df_plot = pd.DataFrame([perc_cost_135, perc_cost_246],index=['Fixed tariff', 'Dynamic tariff'],columns=xlab).transpose()
df_plot.plot(kind='bar',ax=ax)
plt.xticks(rotation=360)
plt.ylabel('Cost deviation from perfect information (DA + BAL) [AUD/kWh]')
plt.tight_layout()
plt.grid()
plt.savefig('figures/total_perc_barchart.png', bbox_inches='tight')

# percentage costs accross scenarios per prosumer -- PLOT
rev_6 = np.array((-cost_DA, -imbal_cost_6, -cost_DA-imbal_cost_6))
rev_6resP = np.array((-cost_resP, -imbal_cost_6resP, -cost_resP -imbal_cost_6resP))
xlab = ['Day-ahead','Balancing','Total']
tot_cost = plt.figure(5,figsize=[10,6])
ax = tot_cost.add_subplot(111)
res = pd.DataFrame([rev_6, rev_6resP],index=['No reserve','With reserve'],columns=xlab)
df_plot = pd.DataFrame([rev_6, rev_6resP],index=['No reserve','With reserve'],columns=xlab)
df_plot.plot(kind='bar',ax=ax)
plt.xticks(rotation=360)
plt.ylabel('Community revenues over the whole year [AUD]')
plt.tight_layout()
plt.grid()
plt.savefig('figures/rev_6-6resP.png', bbox_inches='tight')

#try with scores barchart --PLOT
N = 2
res_scenario_DA = np.array((-cost_DA, -cost_resP))
res_scenario_BAL = np.array((-imbal_cost_6, -imbal_cost_6resP))
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, res_scenario_DA, width, color='#d15448')
p2 = plt.bar(ind, res_scenario_BAL, width,
             bottom=res_scenario_DA)
plt.ylabel('Community revenues over the whole year [AUD]')
#plt.title('Scores by group and gender')
plt.xticks(ind, ('No reserve','With reserve'))
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Day-ahead', 'Balancing'))
plt.savefig('figures/rev_6-6resP.png', bbox_inches='tight')

# reserve distribution among prosumers--PLOT
res_distrib = np.array((-cost_DA, -imbal_cost_6))
rev_6resP = np.array((-cost_resP, -imbal_cost_6resP))
objects = ('0','1','2','3','4','5','6','7','8','9','10','11','12','13','14')
y_pos = np.arange(len(objects))
performance = res_UP_distribution_p
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('UP-regulation reserve allocation over the whole year [kWh]')
plt.ylabel('Prosumer index')
#plt.title('Programming language usage')
plt.savefig('figures/r_up_distribution.png', bbox_inches='tight')

# percentage welfare accross scenarios per prosumer -- PLOT
perc_W_135 = np.array((-WPercentage_1_case1,-WPercentage_3_case1,-WPercentage_5_case1))
perc_W_246 = np.array((-WPercentage_2_case1,-WPercentage_4_case1,-WPercentage_6_case1))
xlab = ['Scenario 1','Scenario 2','Scenario 3']
tot_cost = plt.figure(6,figsize=[10,6])
ax = tot_cost.add_subplot(111)
res = pd.DataFrame([perc_W_135, perc_W_246],index=['Fixed tariff', 'Dynamic tariff'],columns=xlab).transpose()
df_plot = pd.DataFrame([perc_W_135, perc_W_246],index=['Fixed tariff', 'Dynamic tariff'],columns=xlab).transpose()
df_plot.plot(kind='bar',ax=ax)
plt.xticks(rotation=360)
plt.ylabel('Welfare deviation from the perfect information case [%]')
plt.tight_layout()
plt.grid()
plt.savefig('figures/wPerc_s1-6.png', bbox_inches='tight')

plt.figure(10,figsize=[10,6])
plt.plot(np.arange(0,0.02,0.002),R_UP_avrg)
plt.xlabel('Delta-UP [AUD/kWh]')
plt.ylabel('Average UP reserve requirement [kWh]')
plt.savefig('figures/cristo.png', bbox_inches='tight')

fig_percCost_p = plt.figure(10,figsize=[10,6])
ax1, = plt.plot(costPercentage_p_1_case1,label='scenario 1F')
ax2, = plt.plot(costPercentage_p_2_case1,label='scenario 1D')
ax3, = plt.plot(costPercentage_p_3_case1,label='scenario 2F')
ax4, = plt.plot(costPercentage_p_4_case1,label='scenario 2D')
ax5, = plt.plot(costPercentage_p_5_case1,label='scenario 3F',linestyle='dashed')
ax6, = plt.plot(costPercentage_p_6_case1,label='scenario 3D',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4, ax5, ax6], ['scenario 1F', 'scenario 1D','scenario 2F', 'scenario 2D','scenario 3F', 'scenario 3D'])
plt.xlabel('prosumer index')
plt.ylabel('deviation costs [%]')
plt.title('Percentage deviation costs from perfect information per prosumer across scenarios')
plt.savefig('figures/percCost_p_scenarios.png', bbox_inches='tight')

