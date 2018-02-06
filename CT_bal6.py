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
from TMST_def import TMST, TMST_run, n_days

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
# retrieving and extending (1h --> 15min) set points and DA price from DA stage
#==============================================================================

from CT_DA import CT_p_sol, CT_l_sol
p_tilde = CT_p_sol.T
l_tilde = CT_l_sol.T # negative

#==============================================================================
# save OLD and calculate NEW Pmax, Pmin, Lmax, Lmin, intercepts and slopes
# and extend them (they'll be referred to 15 mins time interval)
#==============================================================================
Pmax_DA = Pmax
Pmin_DA = Pmin
Lmax_DA = Lmax
Lmin_DA = Lmin

# new Pmax and Pmin, Lmax and Lmin (referred to 15 mins time interval)----CHECK if correct as a concept!!!
Pmax_bal = np.zeros((TMST_run,n))
Pmin_bal = np.zeros((TMST_run,n))
Lmax_bal = np.zeros((TMST_run,n))
Lmin_bal = np.zeros((TMST_run,n))
for t in range(TMST_run):
    for p in range(n):
        Pmax_bal[t,p] = Pmax_DA[p] + PV[t,p] - p_tilde[t,p] 
        Pmin_bal[t,p] = Pmin_DA[p] + PV[t,p] - p_tilde[t,p]
        Lmax_bal[t,p] = Lmax_DA[t,p] - l_tilde[t,p]
        Lmin_bal[t,p] = Lmin_DA[t,p] - l_tilde[t,p]

#==============================================================================
# COST and UTILITY FUNCTION - shifted
#==============================================================================
y0_g_bal = np.zeros((TMST_run,n))
y0_c_bal = np.zeros((TMST_run,n))

#CONSUMERS
# intercept
for t in range(TMST_run):
    for p in range(n):
        y0_c_bal[t,p] = y0_c[t,p] + mm_c[t,p]*l_tilde[t,p]
# slope
mm_c_bal = mm_c

# GENERATORS        
# intercept  
for t in range(TMST_run):
    for p in range(n):
        y0_g_bal[t,p] = y0_g[t,p] + mm_g[t,p]*p_tilde[t,p]  
# slope
mm_g_bal = mm_g

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

#%% Centralized COMMUNITY BALANCING scenario 6

CT_p_sol_bal6 = np.zeros((g,TMST_run))
CT_l_sol_bal6 = np.zeros((n,TMST_run))
CT_q_sol_bal6 = np.zeros((n,TMST_run))
CT_alfa_sol_bal6 = np.zeros((n,TMST_run))
CT_beta_sol_bal6 = np.zeros((n,TMST_run))
CT_imp_sol_bal6 = np.zeros(TMST_run)
CT_exp_sol_bal6 = np.zeros(TMST_run)
CT_obj_sol_bal6 = np.zeros(TMST_run)
CT_price_sol_bal6 = np.zeros((n,TMST_run))
CT_price2_sol_bal6 = np.zeros((3,TMST_run))

uff = np.int32(24*n_days)
for t in np.arange(0,TMST_run,uff):  # for t = 0, 24
    temp = range(t,t+uff)
    # Create a new model
    CT_m = gb.Model("qp")
    CT_m.setParam( 'OutputFlag', False ) # Quieting Gurobi output
    # Create variables
    p = np.array([CT_m.addVar(lb=Pmin_bal[t,i], ub=Pmax_bal[t,i]) for i in range(n) for k in range(uff)])
    l = np.array([CT_m.addVar(lb=Lmin_bal[t+k,i], ub=Lmax_bal[t+k,i]) for i in range(n) for k in range(uff)])
    q_imp = np.array([CT_m.addVar() for k in range(uff)])
    q_exp = np.array([CT_m.addVar() for k in range(uff)])
    alfa = np.array([CT_m.addVar() for i in range(n) for k in range(uff)])
    beta = np.array([CT_m.addVar() for i in range(n) for k in range(uff)])
    q_pos = np.array([CT_m.addVar() for i in range(n) for k in range(uff)])
    q = np.array([CT_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(uff)]) 
    gamma_i = el_price_UP[temp]     
    gamma_e = -el_price_DW[temp]       
    CT_m.update()
    
    p = np.transpose(p.reshape(n,uff))
    l = np.transpose(l.reshape(n,uff))
    q = np.transpose(q.reshape(n,uff))
    q_pos = np.transpose(q_pos.reshape(n,uff))
    alfa = np.transpose(alfa.reshape(n,uff))
    beta = np.transpose(beta.reshape(n,uff))
    
    # Set objective: 
    obj = (sum(sum(y0_c_bal[temp,:]*l + mm_c_bal[temp,:]/2*l*l) + sum(y0_g_bal[temp,:]*p + mm_g_bal[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(0.001*sum(q_pos)))
    CT_m.setObjective(obj)
    
    # Add constraint
    for k in range(uff):
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
    for k in range(uff):
        for i in range(n):
            CT_price_sol_bal6[i,t+k] = CT_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT_p_sol_bal6[i,t+k] = p[k,i].x
            CT_l_sol_bal6[i,t+k] = l[k,i].x
            CT_q_sol_bal6[i,t+k] = q[k,i].x
            CT_alfa_sol_bal6[i,t+k] = alfa[k,i].x
            CT_beta_sol_bal6[i,t+k] = beta[k,i].x
        CT_price2_sol_bal6[0,t+k] = CT_m.getConstrByName("comm[%s]"%k).Pi
        CT_price2_sol_bal6[1,t+k] = CT_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT_price2_sol_bal6[2,t+k] = CT_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT_imp_sol_bal6[t+k] = q_imp[k].x
        CT_exp_sol_bal6[t+k] = q_exp[k].x
# http://www.gurobi.com/documentation/7.5/refman/attributes.html
    del CT_m

CT_IE_sol_bal6 = CT_imp_sol_bal6 - CT_exp_sol_bal6
