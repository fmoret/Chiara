# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:10:12 2018

@author: Chiara
"""

import os
import gurobipy as gb
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pathlib import Path
import shelve
from TMST_def import TMST, TMST_run

dd1 = os.getcwd()
data_path = str(Path(dd1).parent.parent)+r'\trunk\Input Data 2'
data_path2 = str(Path(dd1).parent.parent)+r'\branches\balancing'

filename=data_path+r'\input_data_2.out'
file_loc=data_path+r'\el_prices.xlsx'

#%%
#==============================================================================
# INPUT DATA - original
#==============================================================================
d = shelve.open(filename, 'r')
el_price_dataframe = pd.read_excel(file_loc)/1000
el_price_notSampled = el_price_dataframe.values
el_price_sampled = el_price_notSampled[1::2] 
el_price = el_price_sampled[:,1]
b2 = d['b2']        
c2 = d['c2']        
b1 = d['b1']
c1 = d['c1']
Pmin = d['Pmin']
Pmax = d['Pmax']
PV = d['PV']       
Load = d['Load']
Flex_load = d['Flex_load']
Agg_load = Load + Flex_load
PostCode = d['PostCode']

#os.chdir(dd1)
n = b2.shape[1]
g = b1.shape[1]

Lmin = - (Agg_load + Flex_load)
Lmax = - Load #baseload
el_price_e = el_price
tau = 0.1
window = 4

#==============================================================================
# COST and UTILITY FUNCTION - original
#==============================================================================
# consumers UTILITY FUNCTION
y0_c = 0.01 + b2 - c2*(Lmax+Lmin)/(Lmax-Lmin)
mm_c = 2*c2/(Lmax-Lmin)
if sum(abs(mm_c[np.isinf(mm_c)]))>0:
    #coefficiente angolare
    y0_c[np.isinf(abs(mm_c))] = 0.01 + b2[np.isinf(abs(mm_c))]  
    #intercetta
    mm_c[np.isinf(abs(mm_c))] = 0                              

# generators COST FUNCTION
y0_g = 0.01 + b1 - c1*(Pmax+Pmin)/(Pmax-Pmin)
mm_g = 2*c1/(Pmax-Pmin)
if sum(abs(mm_g[np.isinf(mm_g)]))>0:
    y0_g[np.isinf(abs(mm_g))] = 0.01 + b1[np.isinf(abs(mm_g))]
    mm_g[np.isinf(abs(mm_g))] = 0
y0_g[np.isnan(y0_g)] = 0
mm_g[np.isnan(mm_g)] = 0

#==============================================================================
# data extension 1h --> 15 mins (repeat each row 4 times)
#==============================================================================
b2_bal = np.repeat(b2, 4, axis=0)
c2_bal = np.repeat(c2, 4, axis=0)
b1_bal = np.repeat(b1, 4, axis=0)
c1_bal = np.repeat(c1, 4, axis=0)
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
# # IMBALANCES
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
# retrieving and extending (1h --> 15min) set points and DA price from DA stage
#==============================================================================
#p_tilde = np.repeat(p_1, 4, axis=0)
#l_tilde = np.repeat(l_1, 4, axis=0)
#lambda_CED = np.repeat(price_community, 4, axis=0)
from CT_res import CT_p_sol_res, CT_l_sol_res, CT_price2_sol_res
p_tilde = np.repeat(CT_p_sol_res.T, 4, axis=0)
l_tilde = np.repeat(CT_l_sol_res.T, 4, axis=0)
lambda_CED = np.repeat(-CT_price2_sol_res[0,:], 4, axis=0)

#==============================================================================
# save OLD and calculate NEW Pmax, Pmin, Lmax, Lmin, intercepts and slopes
# and extend them (they'll be referred to 15 mins time interval)
#==============================================================================
Pmax_DA = Pmax
Pmin_DA = Pmin
Lmax_DA = np.repeat(Lmax, 4, axis=0)
Lmin_DA = np.repeat(Lmin, 4, axis=0)
y0_c_DA = np.repeat(y0_c, 4, axis=0)
y0_g_DA = np.repeat(y0_g, 4, axis=0)
mm_c_DA = np.repeat(mm_c, 4, axis=0)
mm_g_DA = np.repeat(mm_g, 4, axis=0)

# new Pmax and Pmin, Lmax and Lmin (referred to 15 mins time interval)----CHECK if correct as a concept!!!
Pmax_bal = np.zeros((4*TMST_run,n))
Pmin_bal = np.zeros((4*TMST_run,n))
Lmax_bal = np.zeros((4*TMST_run,n))
Lmin_bal = np.zeros((4*TMST_run,n))
for t in range(4*TMST_run):
    for p in range(n):
        Pmax_bal[t,p] = Pmax_DA[p] + PV_bal[t,p] - p_tilde[t,p] 
        Pmin_bal[t,p] = Pmin_DA[p] + PV_bal[t,p] - p_tilde[t,p]
        Lmax_bal[t,p] = Lmax_DA[t,p] - l_tilde[t,p]
        Lmin_bal[t,p] = Lmin_DA[t,p] - l_tilde[t,p]

#==============================================================================
# COST and UTILITY FUNCTION - shifted
#==============================================================================
y0_g_bal = np.zeros((4*TMST,n))
y0_c_bal = np.zeros((4*TMST,n))

#CONSUMERS
# intercept
for t in range(4*TMST_run):
    for p in range(n):
        y0_c_bal[t,p] = y0_c_DA[t,p] + mm_c_DA[t,p]*l_tilde[t,p]
# slope
mm_c_bal = mm_c_DA

# GENERATORS        
# intercept  
for t in range(4*TMST_run):
    for p in range(n):
        y0_g_bal[t,p] = y0_g_DA[t,p] + mm_g_DA[t,p]*p_tilde[t,p]  
# slope
mm_g_bal = mm_g_DA

#==============================================================================
# PRICES
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

#%% Centralized COMMUNITY BALANCING scenario 6

CT_p_sol_bal = np.zeros((g,4*TMST_run))
CT_l_sol_bal = np.zeros((n,4*TMST_run))
CT_q_sol_bal = np.zeros((n,4*TMST_run))
CT_alfa_sol_bal = np.zeros((n,4*TMST_run))
CT_beta_sol_bal = np.zeros((n,4*TMST_run))
CT_imp_sol_bal = np.zeros(4*TMST_run)
CT_exp_sol_bal = np.zeros(4*TMST_run)
CT_obj_sol_bal = np.zeros(4*TMST_run)
CT_price_sol_bal = np.zeros((n,4*TMST_run))
CT_price2_sol_bal = np.zeros((3,4*TMST_run))

for t in np.arange(0,4*TMST_run,96):  # for t = 0, 24
    temp = range(t,t+96)
    # Create a new model
    CT_m = gb.Model("qp")
    CT_m.setParam( 'OutputFlag', False ) # Quieting Gurobi output
    # Create variables
    p = np.array([CT_m.addVar(lb=Pmin_bal[t,i], ub=Pmax_bal[t,i]) for i in range(n) for k in range(96)])
    l = np.array([CT_m.addVar(lb=Lmin_bal[t+k,i], ub=Lmax_bal[t+k,i]) for i in range(n) for k in range(96)])
    q_imp = np.array([CT_m.addVar() for k in range(96)])
    q_exp = np.array([CT_m.addVar() for k in range(96)])
    alfa = np.array([CT_m.addVar() for i in range(n) for k in range(96)])
    beta = np.array([CT_m.addVar() for i in range(n) for k in range(96)])
    q_pos = np.array([CT_m.addVar() for i in range(n) for k in range(96)])
    q = np.array([CT_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(96)]) 
    gamma_i = el_price_UP[temp]     
    gamma_e = -el_price_DW[temp]       
    CT_m.update()
    
    p = np.transpose(p.reshape(n,96))
    l = np.transpose(l.reshape(n,96))
    q = np.transpose(q.reshape(n,96))
    q_pos = np.transpose(q_pos.reshape(n,96))
    alfa = np.transpose(alfa.reshape(n,96))
    beta = np.transpose(beta.reshape(n,96))
    
    # Set objective: 
    obj = (sum(sum(y0_c_bal[temp,:]*l + mm_c_bal[temp,:]/2*l*l) + sum(y0_g_bal[temp,:]*p + mm_g_bal[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(0.001*sum(q_pos)))
    CT_m.setObjective(obj)
    
    # Add constraint
    for k in range(96):
        CT_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k))
        for i in range(n): 
            CT_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] + deltaPV[k,i] - deltaLoad[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT_m.addConstr(q[k,i] <= q_pos[k,i]) 
            CT_m.addConstr(q[k,i] >= -q_pos[k,i])
        CT_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
    #for i in range(n): 
        #CT_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
    CT_m.update()    
        
    CT_m.optimize()
    for k in range(96):
        for i in range(n):
            CT_price_sol_bal[i,t+k] = CT_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT_p_sol_bal[i,t+k] = p[k,i].x
            CT_l_sol_bal[i,t+k] = l[k,i].x
            CT_q_sol_bal[i,t+k] = q[k,i].x
            CT_alfa_sol_bal[i,t+k] = alfa[k,i].x
            CT_beta_sol_bal[i,t+k] = beta[k,i].x
        CT_price2_sol_bal[0,t+k] = CT_m.getConstrByName("comm[%s]"%k).Pi
        CT_price2_sol_bal[1,t+k] = CT_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT_price2_sol_bal[2,t+k] = CT_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT_imp_sol_bal[t+k] = q_imp[k].x
        CT_exp_sol_bal[t+k] = q_exp[k].x
# http://www.gurobi.com/documentation/7.5/refman/attributes.html
    del CT_m

CT_IE_sol_bal = CT_imp_sol_bal - CT_exp_sol_bal
