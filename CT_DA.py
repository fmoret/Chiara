# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:00:37 2018

@author: Chiara
"""
import os
import gurobipy as gb
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pathlib import Path
import shelve
from TMST_def import TMST_run, n_days

dd1 = os.getcwd() #os.path.realpath(__file__) #os.getcwd()
data_path = str(Path(dd1).parent.parent)+r'\trunk\Input Data 2'
data_path2 = str(Path(dd1).parent.parent)+r'\branches\balancing'

filename=data_path+r'\input_data_2.out'
file_loc=data_path+r'\el_prices.xlsx'
#%dd1 = 'M:/PhD/Chiara/branches/balancing'
#%dd2 = 'M:/PhD/Chiara/trunk/Input Data 2'os.chdir(dd1)
#dd1 = 'C:/Users/Chiara/Desktop/special_course/branches/balancing'
#dd2 = 'C:/Users/Chiara/Desktop/special_course/trunk/Input Data 2'
#os.chdir(dd1)

#%% CED - definition of parameters and utility and cost curves
#os.chdir(dd2)
#import shelve
#filename='input_data_2.out'
d = shelve.open(filename, 'r')
#el_price = d['tot_el_price']#[0::2]/1000
el_price_dataframe = pd.read_excel(file_loc)/1000
el_price_notSampled = el_price_dataframe.values
el_price_sampled = el_price_notSampled[1::2] #ogni 2 valori prende il secondo valore
el_price = el_price_sampled[:,1]
b2 = d['b2']        # questi sono i coefficienti dei costi/utilities - quadratic function
c2 = d['c2']        # 2 si riferisce ai consumers, 1 ai generators
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
#TMST = 8760#b2.shape[0]

Lmin = - (Agg_load + Flex_load)
Lmax = - Load #baseload
el_price_e = el_price
tau = 0.1
window = 4

# consumers UTILITY FUNCTION
y0_c = 0.01 + b2 - c2*(Lmax+Lmin)/(Lmax-Lmin)
mm_c = 2*c2/(Lmax-Lmin)
if sum(abs(mm_c[np.isinf(mm_c)]))>0:    # se trova degli inf
    #coefficiente angolare
    y0_c[np.isinf(abs(mm_c))] = 0.01 + b2[np.isinf(abs(mm_c))]  # porta questo a 0.01 + b2
    #intercetta
    mm_c[np.isinf(abs(mm_c))] = 0                               # e questo a 0

# generators COST FUNCTION
y0_g = 0.01 + b1 - c1*(Pmax+Pmin)/(Pmax-Pmin)
mm_g = 2*c1/(Pmax-Pmin)
if sum(abs(mm_g[np.isinf(mm_g)]))>0:
    y0_g[np.isinf(abs(mm_g))] = 0.01 + b1[np.isinf(abs(mm_g))]
    mm_g[np.isinf(abs(mm_g))] = 0
y0_g[np.isnan(y0_g)] = 0        # se non produco il mio costo è 0
mm_g[np.isnan(mm_g)] = 0
     

#%% Community CENTRALIZED solution
#TMST = 8760
CT_p_sol = np.zeros((g,TMST_run))
CT_l_sol = np.zeros((n,TMST_run))
CT_q_sol = np.zeros((n,TMST_run))
CT_alfa_sol = np.zeros((n,TMST_run))
CT_beta_sol = np.zeros((n,TMST_run))
CT_imp_sol = np.zeros(TMST_run)
CT_exp_sol = np.zeros(TMST_run)
CT_obj_sol = np.zeros(TMST_run)
CT_price_sol = np.zeros((n,TMST_run))
CT_price2_sol = np.zeros((3,TMST_run))
uff = np.int32(24*n_days)
for t in np.arange(0,TMST_run,uff):  # for t = 0, 24
    temp = range(t,t+uff)
    # Create a new model
    CT_m = gb.Model("qp")
    #CT_m.setParam( 'OutputFlag', False ) # Quieting Gurobi output
    # Create variables
    p = np.array([CT_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(uff)])
    l = np.array([CT_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(uff)])
    q_imp = np.array([CT_m.addVar() for k in range(uff)])
    q_exp = np.array([CT_m.addVar() for k in range(uff)])
    alfa = np.array([CT_m.addVar() for i in range(n) for k in range(uff)])
    beta = np.array([CT_m.addVar() for i in range(n) for k in range(uff)])
    q_pos = np.array([CT_m.addVar() for i in range(n) for k in range(uff)])
    q = np.array([CT_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(uff)]) 
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    CT_m.update()
    
    p = np.transpose(p.reshape(n,uff))
    l = np.transpose(l.reshape(n,uff))
    q = np.transpose(q.reshape(n,uff))
    q_pos = np.transpose(q_pos.reshape(n,uff))
    alfa = np.transpose(alfa.reshape(n,uff))
    beta = np.transpose(beta.reshape(n,uff))
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(0.001*sum(q_pos)))

    CT_m.setObjective(obj)
    
    # Add constraint
    for k in range(uff):
        CT_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k)) #somma sui prosumer
        #what about mettere una variabile invece che 0? da minimizzare poi nell'obj
        for i in range(n): #balance per ogni prosumer, ogni ora
            CT_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT_m.addConstr(q[k,i] <= q_pos[k,i]) #limite sulla q, che può essere pos o neg
            CT_m.addConstr(q[k,i] >= -q_pos[k,i])
        CT_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
    
    if n_days == 1:
        for i in range(n): 
            CT_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
        CT_m.update()    
        
    CT_m.optimize()
    for k in range(uff):
        for i in range(n):
            CT_price_sol[i,t+k] = CT_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT_p_sol[i,t+k] = p[k,i].x
            CT_l_sol[i,t+k] = l[k,i].x
            CT_q_sol[i,t+k] = q[k,i].x
            CT_alfa_sol[i,t+k] = alfa[k,i].x
            CT_beta_sol[i,t+k] = beta[k,i].x
        CT_price2_sol[0,t+k] = CT_m.getConstrByName("comm[%s]"%k).Pi
        CT_price2_sol[1,t+k] = CT_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT_price2_sol[2,t+k] = CT_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT_imp_sol[t+k] = q_imp[k].x
        CT_exp_sol[t+k] = q_exp[k].x
# http://www.gurobi.com/documentation/7.5/refman/attributes.html
    del CT_m

CT_IE_sol = CT_imp_sol - CT_exp_sol


W_DA_tp = np.zeros([TMST_run,n]) 
for t in range(TMST_run):
    for p in range(n):
        W_DA_tp[t,p] = CT_alfa_sol[p,t]*(el_price_e[t]+0.1) + CT_beta_sol[p,t]*(-el_price_e[t]) +\
        y0_c[t,p]*CT_l_sol[p,t] + mm_c[t,p]/2*CT_l_sol[p,t]*CT_l_sol[p,t] + y0_g[t,p]*CT_p_sol[p,t] + mm_g[t,p]/2*CT_p_sol[p,t]*CT_p_sol[p,t]+\
                           + (-CT_price2_sol[0,t])*(CT_q_sol[p,t])

W_DA_p = np.sum(W_DA_tp, axis=0)
W_DA = np.sum(W_DA_p)

cost_DA_tp = np.zeros([TMST_run,n]) 
for t in range(TMST_run):
    for p in range(n):
        cost_DA_tp[t,p] = + (-CT_price2_sol[0,t])*(CT_q_sol[p,t]) + CT_alfa_sol[p,t]*(el_price_e[t]+0.1) + CT_beta_sol[p,t]*(-el_price_e[t])

#cost_DA_tp_times4 = np.repeat(cost_DA_tp, 4, axis=0)
cost_DA_p = np.sum(cost_DA_tp, axis=0)
cost_DA = np.sum(cost_DA_p)
