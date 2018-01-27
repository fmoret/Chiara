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
#import matplotlib.pyplot as plt
from pathlib import Path
import shelve

dd1 = os.getcwd() #os.path.realpath(__file__) #os.getcwd()
data_path = str(Path(dd1).parent.parent)+r'\trunk\Input Data 2'
data_path2 = str(Path(dd1).parent.parent)+r'\branches\balancing'

filename=data_path+r'\input_data_2.out'
file_loc=data_path+r'\el_prices.xlsx'

from CT_DA import cost_DA, cost_DA_p
from CT_DApi import cost_DA_PI1_p, cost_DA_PI1
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
# data extension 1h --> 15 mins (repeat each row 4 times)
#==============================================================================

PV_bal = np.repeat(PV, 4, axis=0)
Load_bal = np.repeat(Load, 4, axis=0)
Flex_load_bal = np.repeat(Flex_load, 4, axis=0)

#==============================================================================
# create NOISE
#==============================================================================

# create noise for PV (only if PV nonzero) 
noise_PV_DA_dataframe = pd.read_csv(data_path2+r'\noise_PV_DA.csv', header=None)
noise_PV_DA = noise_PV_DA_dataframe.values
for row in range(noise_PV_DA.shape[0]):
    for col in range(noise_PV_DA.shape[1]):
        if PV[row,col] == 0:
            noise_PV_DA[row,col] = 0            
# adding noise to PV (and if the result is less than zero bring it to zero)
PV_real_DA = PV + noise_PV_DA
for row in range(PV_real_DA.shape[0]):
    for col in range(PV_real_DA.shape[1]):
        if PV_real_DA[row,col] < 0:
            PV_real_DA[row,col] = 0
PV_real_bal = np.repeat(PV_real_DA, 4, axis=0)
# create noise for Load
noise_Load_DA_dataframe = pd.read_csv(data_path2+r'\noise_Load_DA.csv', header=None)
noise_Load_DA = noise_Load_DA_dataframe.values
# adding noise to Load (and if the result is less than zero bring it to zero)
Load_real_DA = Load + noise_Load_DA
for row in range(Load_real_DA.shape[0]):
    for col in range(Load_real_DA.shape[1]):
        if Load_real_DA[row,col] < 0:
            Load_real_DA[row,col] = 0
Load_real_bal = np.repeat(Load_real_DA, 4, axis=0)

#==============================================================================
# IMBALANCES
#==============================================================================
deltaPV = PV_real_bal - PV_bal
deltaPV_DA = deltaPV[1::4]
deltaLoad = Load_real_bal - Load_bal
deltaLoad_DA = deltaPV[1::4]
deltaPV_prosumer = np.empty([n])
deltaLoad_prosumer = np.empty([n])
deltaPV_community = np.empty([4*TMST])
deltaLoad_community = np.empty([4*TMST])

for p in range(n):
    deltaPV_prosumer[p] = np.sum(deltaPV[:,p])
    deltaLoad_prosumer[p] = np.sum(deltaLoad[:,p])
for t in range(4*TMST):
    deltaPV_community[t] = np.sum(deltaPV[t,:])
    deltaLoad_community[t] = np.sum(deltaLoad[t,:])
    
imbalance_prosumer = deltaPV_prosumer - deltaLoad_prosumer
imbalance_community = deltaPV_community - deltaLoad_community
average_imbal_community = np.average(imbalance_community)

#==============================================================================
# # BAL PRICES
#==============================================================================
el_price_DA = np.repeat(el_price_e, 4, axis=0)
el_price_DW = np.repeat(el_price_sampled[:,0], 4, axis=0) # 2 rows
el_price_UP = np.repeat(el_price_sampled[:,2], 4, axis=0) # 2 rows

system_state = np.empty([4*TMST])
for t in range(4*TMST):
    if el_price_DA[t] == el_price_DW[t]: #up-regulation
        system_state[t] = 1
    elif el_price_DA[t] == el_price_UP[t]: #dw-regulation
        system_state[t] = 2
    elif el_price_DW[t] == el_price_UP[t]: #balance
        system_state = 0

el_price_BAL = np.empty([4*TMST]) # 1 row 
for t in range(4*TMST):
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
imbal_cost_tp_1 = np.zeros([4*TMST,n])
for t in range(4*TMST):
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
    costPercentage_p_1_case1[p] = (cost_DA_p[p]+imbal_cost_p_1[p]-cost_DA_PI1_p[p])/cost_DA_PI1_p[p]
costPercentage_1_case1 = (cost_DA + imbal_cost_1 - cost_DA_PI1)/cost_DA_PI1
# QoE
numerator_1 = -imbal_cost_tp_1
denominator_1 = deltaPV - deltaLoad
perceived_price_1 = np.sum(numerator_1, axis=0)/np.sum(denominator_1, axis=0)
sigma_1 = np.std(perceived_price_1)
sigmaMax_1 = max(perceived_price_1) - min(perceived_price_1)
QoE_1 = 1 - sigma_1/sigmaMax_1

#%% SCENARIO 2 

# imbalance costs
imbal_cost_tp_2 = np.zeros([4*TMST,n])
for t in range(4*TMST):
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
    costPercentage_p_2_case1[p] = (cost_DA_p[p]+imbal_cost_p_2[p]-cost_DA_PI1_p[p])/cost_DA_PI1_p[p]
costPercentage_2_case1 = (cost_DA + imbal_cost_2 - cost_DA_PI1)/cost_DA_PI1
# QoE
numerator_2 = -imbal_cost_tp_2
denominator_2 = deltaPV - deltaLoad
perceived_price_2 = np.sum(numerator_2, axis=0)/np.sum(denominator_2, axis=0)
sigma_2 = np.std(perceived_price_2)
sigmaMax_2 = max(perceived_price_2) - min(perceived_price_2)
QoE_2 = 1 - sigma_2/sigmaMax_2

#%% SCENARIO 3

# imbalance costs
imbal_cost_t_3 = np.zeros([4*TMST])
for t in range(4*TMST):
    if imbalance_community[t] < 0:
        imbal_cost_t_3[t] = - (ret_price_imp)*(imbalance_community[t])
    else:
        imbal_cost_t_3[t] = - (ret_price_exp)*(imbalance_community[t])
imbal_cost_tp_3 = np.ones([4*TMST,n])
for t in range(4*TMST):
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
    costPercentage_p_3_case1[p] = (cost_DA_p[p]+imbal_cost_p_3[p]-cost_DA_PI1_p[p])/cost_DA_PI1_p[p]
costPercentage_3_case1 = (cost_DA + imbal_cost_3 - cost_DA_PI1)/cost_DA_PI1
# QoE
numerator_3 = -imbal_cost_tp_3
denominator_3 = deltaPV - deltaLoad
perceived_price_3 = np.sum(numerator_3, axis=0)/np.sum(denominator_3, axis=0)
sigma_3 = np.std(perceived_price_3)
sigmaMax_3 = max(perceived_price_3) - min(perceived_price_3)
QoE_3 = 1 - sigma_3/sigmaMax_3

#%% SCENARIO 4

# imbalance costs
imbal_cost_t_4 = np.zeros([4*TMST])
for t in range(4*TMST):
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
imbal_cost_tp_4 = np.ones([4*TMST,n])
for t in range(4*TMST):
    for p in range(n):
        imbal_cost_tp_4[t,p] = imbal_cost_t_4[t]/n
imbal_cost_p_4 = np.sum(imbal_cost_tp_4, axis = 0)
imbal_cost_4 = np.sum(imbal_cost_t_4)
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_4_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_4_case1[p] = (cost_DA_p[p]+imbal_cost_p_4[p]-cost_DA_PI1_p[p])/cost_DA_PI1_p[p]
costPercentage_4_case1 = (cost_DA + imbal_cost_4 - cost_DA_PI1)/cost_DA_PI1
# QoE
numerator_4 = -imbal_cost_tp_4
denominator_4 = deltaPV - deltaLoad
perceived_price_4 = np.sum(numerator_4, axis=0)/np.sum(denominator_4, axis=0)
sigma_4 = np.std(perceived_price_4)
sigmaMax_4 = max(perceived_price_4) - min(perceived_price_4)
QoE_4 = 1 - sigma_4/sigmaMax_4

#%% SCENARIO 5

from CT_bal5 import (CT_price2_sol_bal, CT_q_sol_bal, CT_beta_sol_bal, 
CT_alfa_sol_bal, CT_l_sol_bal, CT_p_sol_bal, mm_c_bal, mm_g_bal, y0_c_bal, y0_g_bal)
# imbalance costs
imbal_cost_tp_5 = np.empty([4*TMST_run,n])
numerator_5 = np.empty([4*TMST_run,n])
denominator_5 = np.empty([4*TMST_run,n])
for t in range(4*TMST_run):
    for p in range(n):
        imbal_cost_tp_5[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*ret_price_exp + CT_alfa_sol_bal[p,t]*(ret_price_imp) + \
        y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        numerator_5[t,p] = -(-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) + CT_beta_sol_bal[p,t]*ret_price_exp - CT_alfa_sol_bal[p,t]*ret_price_imp
        denominator_5[t,p] = CT_beta_sol_bal[p,t] - CT_alfa_sol_bal[p,t] - CT_q_sol_bal[p,t]
imbal_cost_p_5 = np.sum(imbal_cost_tp_5, axis=0)
imbal_cost_5 = np.sum(imbal_cost_p_5)
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_5_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_5_case1[p] = (cost_DA_p[p]+imbal_cost_p_5[p]-cost_DA_PI1_p[p])/cost_DA_PI1_p[p]
costPercentage_5_case1 = (cost_DA + imbal_cost_5 - cost_DA_PI1)/cost_DA_PI1
# QoE
perceived_price_5 = np.sum(numerator_5, axis = 0)/np.sum(denominator_5, axis = 0)
sigma_5 = np.std(perceived_price_5)
sigmaMax_5 = max(perceived_price_5) - min(perceived_price_5)
QoE_5 = 1 - sigma_5/sigmaMax_5

#%% SCENARIO 6
from CT_bal6 import (CT_price2_sol_bal, CT_q_sol_bal, CT_beta_sol_bal, 
CT_alfa_sol_bal, CT_l_sol_bal, CT_p_sol_bal, mm_c_bal, mm_g_bal, y0_c_bal, y0_g_bal)
# imbalance costs
imbal_cost_tp_6 = np.empty([4*TMST_run,n])
numerator_6 = np.empty([4*TMST_run,n])
denominator_6 = np.empty([4*TMST_run,n])
for t in range(4*TMST_run):
    for p in range(n):
        if system_state[t] == 2:
            imbal_cost_tp_6[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DW[t] + CT_alfa_sol_bal[p,t]*el_price_DA[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        elif system_state[t] == 1:
            imbal_cost_tp_6[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DA[t] + CT_alfa_sol_bal[p,t]*el_price_UP[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        else:
            imbal_cost_tp_6[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DA[t] + CT_alfa_sol_bal[p,t]*el_price_DA[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        numerator_6[t,p] = -(-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) + CT_beta_sol_bal[p,t]*el_price_DW[t] - CT_alfa_sol_bal[p,t]*el_price_UP[t]
        denominator_6[t,p] = CT_beta_sol_bal[p,t] - CT_alfa_sol_bal[p,t] - CT_q_sol_bal[p,t]
imbal_cost_p_6= np.sum(imbal_cost_tp_6, axis=0)
imbal_cost_6= np.sum(imbal_cost_p_6) # just to check that this is equal to imbal_cost_6
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_6_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_6_case1[p] = (cost_DA_p[p]+imbal_cost_p_6[p]-cost_DA_PI1_p[p])/cost_DA_PI1_p[p]
costPercentage_6_case1 = (cost_DA + imbal_cost_6 - cost_DA_PI1)/cost_DA_PI1
# QoE
perceived_price_6 = np.sum(numerator_6, axis = 0)/np.sum(denominator_6, axis = 0)
sigma_6 = np.std(perceived_price_6)
sigmaMax_6 = max(perceived_price_6) - min(perceived_price_6)
QoE_6 = 1 - sigma_6/sigmaMax_6

#%% SCENARIO 6 - res
from CT_bal6res import (CT_price2_sol_bal, CT_q_sol_bal, CT_beta_sol_bal, 
CT_alfa_sol_bal, CT_l_sol_bal, CT_p_sol_bal, mm_c_bal, mm_g_bal, y0_c_bal, y0_g_bal)
from CT_res import cost_res_p, cost_res
# imbalance costs
imbal_cost_tp_6res = np.empty([4*TMST_run,n])
numerator_6res = np.empty([4*TMST_run,n])
denominator_6res = np.empty([4*TMST_run,n])
for t in range(4*TMST_run):
    for p in range(n):
        if system_state[t] == 2:
            imbal_cost_tp_6res[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DW[t] + CT_alfa_sol_bal[p,t]*el_price_DA[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        elif system_state[t] == 1:
            imbal_cost_tp_6res[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DA[t] + CT_alfa_sol_bal[p,t]*el_price_UP[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        else:
            imbal_cost_tp_6res[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DA[t] + CT_alfa_sol_bal[p,t]*el_price_DA[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        numerator_6res[t,p] = -(-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) + CT_beta_sol_bal[p,t]*el_price_DW[t] - CT_alfa_sol_bal[p,t]*el_price_UP[t]
        denominator_6res[t,p] = CT_beta_sol_bal[p,t] - CT_alfa_sol_bal[p,t] - CT_q_sol_bal[p,t]
imbal_cost_p_6res= np.sum(imbal_cost_tp_6res, axis=0)
imbal_cost_6res= np.sum(imbal_cost_p_6res)
imbal_cost_p_perunit_imbal_6res = -abs(imbal_cost_p_6res)/imbalance_prosumer
# QoE
perceived_price_6res = np.sum(numerator_6res, axis = 0)/np.sum(denominator_6res, axis = 0)
sigma_6res = np.std(perceived_price_6res)
sigmaMax_6res = max(perceived_price_6res) - min(perceived_price_6res)
QoE_6res = 1 - sigma_6res/sigmaMax_6res
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_6res_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_6res_case1[p] = (cost_res_p[p]+imbal_cost_p_6res[p]-cost_DA_PI1_p[p])/cost_DA_PI1_p[p]
costPercentage_6res_case1 = (cost_res + imbal_cost_6res - cost_DA_PI1)/cost_DA_PI1

#%% SCENARIO 6 - resP
from CT_bal6resP import (CT_price2_sol_bal, CT_q_sol_bal, CT_beta_sol_bal, 
CT_alfa_sol_bal, CT_l_sol_bal, CT_p_sol_bal, mm_c_bal, mm_g_bal, y0_c_bal, y0_g_bal)
from CT_resP import cost_resP_p, cost_resP
# imbalance costs
imbal_cost_tp_6resP = np.empty([4*TMST_run,n])
numerator_6resP = np.empty([4*TMST_run,n])
denominator_6resP = np.empty([4*TMST_run,n])
for t in range(4*TMST_run):
    for p in range(n):
        if system_state[t] == 2:
            imbal_cost_tp_6resP[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DW[t] + CT_alfa_sol_bal[p,t]*el_price_DA[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        elif system_state[t] == 1:
            imbal_cost_tp_6resP[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DA[t] + CT_alfa_sol_bal[p,t]*el_price_UP[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        else:
            imbal_cost_tp_6resP[t,p] = (-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DA[t] + CT_alfa_sol_bal[p,t]*el_price_DA[t] + \
            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
        numerator_6resP[t,p] = -(-CT_price2_sol_bal[0,t])*(CT_q_sol_bal[p,t]) + CT_beta_sol_bal[p,t]*el_price_DW[t] - CT_alfa_sol_bal[p,t]*el_price_UP[t]
        denominator_6resP[t,p] = CT_beta_sol_bal[p,t] - CT_alfa_sol_bal[p,t] - CT_q_sol_bal[p,t]
imbal_cost_p_6resP= np.sum(imbal_cost_tp_6resP, axis=0)
imbal_cost_6resP= np.sum(imbal_cost_p_6resP)
# cost/revenue for each prosumer compared to the Perfect Information
costPercentage_p_6resP_case1 = np.zeros(n)
for p in range(n):
    costPercentage_p_6resP_case1[p] = (cost_resP_p[p]+imbal_cost_p_6resP[p]-cost_DA_PI1_p[p])/cost_DA_PI1_p[p]
costPercentage_6resP_case1 = (cost_resP + imbal_cost_6resP - cost_DA_PI1)/cost_DA_PI1
# QoE
perceived_price_6resP = np.sum(numerator_6resP, axis = 0)/np.sum(denominator_6resP, axis = 0)
sigma_6resP = np.std(perceived_price_6resP)
sigmaMax_6resP = max(perceived_price_6resP) - min(perceived_price_6resP)
QoE_6resP = 1 - sigma_6resP/sigmaMax_6resP

#%% save results in DataFrame

percentage_costs_p = pd.DataFrame(np.array([costPercentage_p_1_case1,costPercentage_p_2_case1,costPercentage_p_3_case1,costPercentage_p_4_case1,costPercentage_p_5_case1,costPercentage_p_6_case1,costPercentage_p_6res_case1,costPercentage_p_6resP_case1]).T, columns=['costPercentage s1','costPercentage s2','costPercentage s3','costPercentage s4','costPercentage s5','costPercentage s6','costPercentage s6res','costPercentage s6resP'])
percentage_costs = pd.DataFrame(np.array([costPercentage_1_case1,costPercentage_2_case1,costPercentage_3_case1,costPercentage_4_case1,costPercentage_5_case1,costPercentage_6_case1,costPercentage_6res_case1,costPercentage_6resP_case1]).reshape(1,-1), columns=['costPercentage s1','costPercentage s2','costPercentage s3','costPercentage s4','costPercentage s5','costPercentage s6','costPercentage s6res','costPercentage s6resP'])

imbal_costs = pd.DataFrame(np.array([imbal_cost_1,imbal_cost_2,imbal_cost_3,imbal_cost_4,imbal_cost_5,imbal_cost_6,imbal_cost_6res,imbal_cost_6resP]).reshape(1,-1), columns=['imbal cost s1','imbal cost s2','imbal cost s3','imbal cost s4','imbal cost s5','imbal cost s6','imbal cost s6res','imbal cost s6resP'])
imbal_costs_p = pd.DataFrame(np.array([imbal_cost_p_1,imbal_cost_p_2,imbal_cost_p_3,imbal_cost_p_4,imbal_cost_p_5,imbal_cost_p_6,imbal_cost_p_6res,imbal_cost_p_6resP]).T, columns=['imbal cost s1','imbal cost s2','imbal cost s3','imbal cost s4','imbal cost s5','imbal cost s6','imbal cost s6res','imbal cost s6resP'])

QoE = pd.DataFrame(np.array([QoE_1,QoE_2,QoE_3,QoE_4,QoE_5,QoE_6,QoE_6res,QoE_6resP]).reshape(1,-1), columns=['imbal cost s1','imbal cost s2','imbal cost s3','imbal cost s4','imbal cost s5','imbal cost s6','imbal cost s6res','imbal cost s6resP'])
