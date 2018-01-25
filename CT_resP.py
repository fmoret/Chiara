# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:04:17 2018

@author: Chiara
"""
import numpy as np
import gurobipy as gb
import shelve
import pandas as pd
from pathlib import Path
import os

dd1 = os.getcwd() 
data_path = str(Path(dd1).parent.parent)+r'\trunk\Input Data 2'
data_path2 = str(Path(dd1).parent.parent)+r'\branches\balancing'

filename=data_path+r'\input_data_2.out'
file_loc=data_path+r'\el_prices.xlsx'

#%% input data - DA (hours) + reserve (pop out)

d = shelve.open(filename, 'r')
el_price_dataframe = pd.read_excel(file_loc)/1000 # dataframe: AUD per kWh ogni mezz'ora
el_price_notSampled = el_price_dataframe.values # only values
el_price_sampled = el_price_notSampled[1::2] # ogni 2 valori prende il secondo valore
el_price = el_price_sampled[:,1] # colonna DA
b2 = d['b2']        # (8760, 15)
c2 = d['c2']        # 2: consumers, 1: generators
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
TMST = 8760#b2.shape[0]

Lmin = - (Agg_load + Flex_load)
Lmax = - Load #baseload
el_price_e = el_price
tau = 0.1
window = 4

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
     
# balancing prices (every 15 mins)
el_price_DA = np.repeat(el_price_e, 4, axis=0)
el_price_DW = np.repeat(el_price_sampled[:,0], 4, axis=0) # 2 rows
el_price_UP = np.repeat(el_price_sampled[:,2], 4, axis=0) # 2 rows

system_state = np.empty([TMST])
for t in range(TMST):
    if el_price_DA[t] == el_price_DW[t]: #up-regulation
        system_state[t] = 1
    elif el_price_DA[t] == el_price_UP[t]: #dw-regulation
        system_state[t] = 2
    elif el_price_DW[t] == el_price_UP[t]: #balance
        system_state = 0

el_price_BAL = np.empty([TMST]) # 1 row (FINAL BAL price)
for t in range(TMST):
    if system_state[t] == 0:                # 0 is balanced
        el_price_BAL[t] = el_price_UP[t]
    elif system_state[t] == 1:              # 1 is UP
        el_price_BAL[t] = el_price_UP[t]
    elif system_state[t] == 2:              # 2 is DW
        el_price_BAL[t] = el_price_DW[t]
    
ret_price_exp = np.average(el_price_DW)
ret_price_imp = np.average(el_price_UP)


#%% Community CENTRALIZED solution with reserve POP-OUT
n = 15
TMST = 48

# prices
delta_lambda_UP = np.average(el_price_UP) - np.average(el_price_e)
delta_lambda_DW = np.average(el_price_DW) - np.average(el_price_e)
lambda_UP = (el_price_e + 0.1) + delta_lambda_UP*np.ones(len(el_price_e))
lambda_DW = (el_price_e) + delta_lambda_DW*np.ones(len(el_price_e))

# allocate variables
CT_p_sol_resP = np.zeros((n,TMST))
CT_l_sol_resP = np.zeros((n,TMST))
CT_q_sol_resP = np.zeros((n,TMST))
CT_R_UP_sol_resP = np.zeros(TMST)
CT_R_DW_sol_resP = np.zeros(TMST)
CT_r_p_UP_sol_resP = np.zeros((n,TMST))
CT_r_p_DW_sol_resP = np.zeros((n,TMST))
CT_r_l_UP_sol_resP = np.zeros((n,TMST))
CT_r_l_DW_sol_resP = np.zeros((n,TMST))
CT_alfa_sol_resP = np.zeros((n,TMST))
CT_beta_sol_resP = np.zeros((n,TMST))
CT_imp_sol_resP = np.zeros(TMST)
CT_exp_sol_resP = np.zeros(TMST)
CT_obj_sol_resP = np.zeros(TMST)
CT_price_sol_resP = np.zeros((n,TMST))
CT_price2_sol_resP = np.zeros((3,TMST))

for t in np.arange(0,TMST,24):  # for t = 0, 24
    temp = range(t,t+24)
    # Create a new model
    CT_m_resP = gb.Model("qp")
    #CT_m.setParam( 'OutputFlag', False ) # Quieting Gurobi output
    # Create variables
    p = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    l = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    r_p_UP = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    r_p_DW = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    r_l_UP = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    r_l_DW = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    R_UP = np.array([CT_m_resP.addVar() for k in range(24)])
    R_DW = np.array([CT_m_resP.addVar() for k in range(24)])
    q_imp = np.array([CT_m_resP.addVar() for k in range(24)])
    q_exp = np.array([CT_m_resP.addVar() for k in range(24)])
    alfa = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    beta = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    q_pos = np.array([CT_m_resP.addVar() for i in range(n) for k in range(24)])
    q = np.array([CT_m_resP.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)]) 
    gamma_res_i = lambda_UP[temp]
    gamma_res_e = -lambda_DW[temp]
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    CT_m_resP.update()
    
    p = np.transpose(p.reshape(n,24))
    l = np.transpose(l.reshape(n,24))
    q = np.transpose(q.reshape(n,24))
    r_p_UP = np.transpose(r_p_UP.reshape(n,24))
    r_p_DW = np.transpose(r_p_DW.reshape(n,24))
    r_l_UP = np.transpose(r_l_UP.reshape(n,24))
    r_l_DW = np.transpose(r_l_DW.reshape(n,24))
    q_pos = np.transpose(q_pos.reshape(n,24))
    alfa = np.transpose(alfa.reshape(n,24))
    beta = np.transpose(beta.reshape(n,24))
    
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*(-l+r_l_UP-r_l_DW) + mm_c[temp,:]/2*(-l+r_l_UP-r_l_DW)*(-l+r_l_UP-r_l_DW)) +\
              sum(y0_g[temp,:]*(p+r_p_UP-r_p_DW) + mm_g[temp,:]/2*(p+r_p_UP-r_p_DW)*(p+r_p_UP-r_p_DW)) +\
              sum(gamma_i*q_imp + gamma_e*q_exp) +\
              sum(0.001*sum(q_pos)) +\
              -sum(gamma_res_i*R_UP + gamma_res_e*R_DW)))
    CT_m_resP.setObjective(obj)
    
    # Add constraint
    for k in range(24):
        for i in range(n):
            CT_m_resP.addConstr(p[k,i] - l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT_m_resP.addConstr(q[k,i] <= q_pos[k,i])
            CT_m_resP.addConstr(q[k,i] >= -q_pos[k,i])
            CT_m_resP.addConstr(p[k,i] - PV[t+k,i] + r_p_UP[k,i] <= Pmax[i], name="R_UP_limit_p[%s,%s]"% (k,i))
            CT_m_resP.addConstr(p[k,i] - PV[t+k,i] - r_p_DW[k,i] >= Pmin[i], name="R_DW_limit_p[%s,%s]"% (k,i))
            CT_m_resP.addConstr(-l[k,i] + r_l_UP[k,i] <= Lmax[t+k,i], name="R_UP_limit_l[%s,%s]"% (k,i))
            CT_m_resP.addConstr(-l[k,i] - r_l_DW[k,i] >= Lmin[t+k,i], name="R_DW_limit_l[%s,%s]"% (k,i))
        CT_m_resP.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k))
        CT_m_resP.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT_m_resP.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
        CT_m_resP.addConstr(sum(r_p_UP[k,:])+sum(r_l_UP[k,:]) - R_UP[k] == 0, name="r_UP_requirement[%s]"%(k))
        CT_m_resP.addConstr(sum(r_p_DW[k,:])+sum(r_l_DW[k,:]) - R_DW[k] == 0, name="r_DW_requirement[%s]"%(k))
        
    for i in range(n): 
        CT_m_resP.addConstr(sum(Agg_load[temp,i] - l[:,i]) == 0) #changed the sign, after changing l from neg to pos
    CT_m_resP.update()    
        
    CT_m_resP.optimize()
    for k in range(24):
        for i in range(n):
            CT_price_sol_resP[i,t+k] = CT_m_resP.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT_p_sol_resP[i,t+k] = p[k,i].x
            CT_l_sol_resP[i,t+k] = l[k,i].x
            CT_q_sol_resP[i,t+k] = q[k,i].x
            CT_alfa_sol_resP[i,t+k] = alfa[k,i].x
            CT_beta_sol_resP[i,t+k] = beta[k,i].x
            CT_r_p_UP_sol_resP[i,t+k] = r_p_UP[k,i].x
            CT_r_p_DW_sol_resP[i,t+k] = r_p_DW[k,i].x
            CT_r_l_UP_sol_resP[i,t+k] = r_l_UP[k,i].x
            CT_r_l_DW_sol_resP[i,t+k] = r_l_DW[k,i].x
        CT_R_UP_sol_resP[t+k] = R_UP[k].x
        CT_R_DW_sol_resP[t+k] = R_DW[k].x    
        CT_price2_sol_resP[0,t+k] = CT_m_resP.getConstrByName("comm[%s]"%k).Pi
        CT_price2_sol_resP[1,t+k] = CT_m_resP.getConstrByName("imp_bal[%s]"%k).Pi
        CT_price2_sol_resP[2,t+k] = CT_m_resP.getConstrByName("exp_bal[%s]"%k).Pi
        CT_imp_sol_resP[t+k] = q_imp[k].x
        CT_exp_sol_resP[t+k] = q_exp[k].x
# http://www.gurobi.com/documentation/7.5/refman/attributes.html
    del CT_m_resP

CT_IE_sol_resP = CT_imp_sol_resP - CT_exp_sol_resP