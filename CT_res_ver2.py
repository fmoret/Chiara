# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:27:14 2018

@author: Chiara
"""
import numpy as np
import gurobipy as gb
import shelve
import pandas as pd
from pathlib import Path
import os
from TMST_def import TMST, TMST_run, n_days

dd1 = os.getcwd()
data_path = str(Path(dd1).parent.parent)+r'\trunk\Input Data 2'
data_path2 = str(Path(dd1).parent.parent)+r'\branches\balancing'

filename=data_path+r'\input_data_2.out'
file_loc=data_path+r'\el_prices.xlsx'


#%% input data - DA (hours) + reserve (uncertainty)

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


n = b2.shape[1]
g = b1.shape[1]


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

#%% Imbalances - all this part needed if we want a correlation between the reserve requirement and the actual imbalance

PV_bal = np.repeat(PV, 4, axis=0)
Load_bal = np.repeat(Load, 4, axis=0)

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

# IMBALANCES
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

#%% Community CENTRALIZED DA + reserve (uncertainty)
n = 15
RES_UP = 0.8*np.ones(8760)
#RES_UP = np.zeros(8760)
#RES_UP = abs(imbalance_community)[1::4]        # reserve requirement - set depending on the average imbalance of the community each hour (dato temporaneo)
RES_DW = RES_UP
CR_UP = 0.005*np.ones([TMST_run,n])
CR_DW = 0.005*np.ones([TMST_run,2*n])
pi_res_UP = 0.05
pi_res_DW = 0.05

CT_p_sol_res = np.zeros((n,TMST_run))
CT_l_sol_res = np.zeros((n,TMST_run))
CT_q_sol_res = np.zeros((n,TMST_run))
CT_r_UP_sol_res = np.zeros((2*n,TMST_run))
CT_r_DW_sol_res = np.zeros((2*n,TMST_run))
CT_R_UP_sol_res = np.zeros((2*n,TMST_run))
CT_R_DW_sol_res = np.zeros((2*n,TMST_run))
CT_alfa_sol_res = np.zeros((n,TMST_run))
CT_beta_sol_res = np.zeros((n,TMST_run))
CT_imp_sol_res = np.zeros(TMST_run)
CT_exp_sol_res = np.zeros(TMST_run)
CT_obj_sol_res = np.zeros(TMST_run)
CT_price_sol_res = np.zeros((n,TMST_run))
CT_price2_sol_res = np.zeros((3,TMST_run))

uff = np.int32(24*n_days)
for t in np.arange(0,TMST_run,uff):  # for t = 0, 24
    temp = range(t,t+uff)
    # Create a new model
    CT_m_res = gb.Model("qp")
    #CT_m.setParam( 'OutputFlag', False ) # Quieting Gurobi output
    # Create variables
    p = np.array([CT_m_res.addVar() for i in range(n) for k in range(uff)])
    l = np.array([CT_m_res.addVar() for i in range(n) for k in range(uff)])
    #r_UP = np.array([CT_m_res.addVar() for i in range(2*n) for k in range(uff)])
    r_UP = np.array([CT_m_res.addVar() for i in range(n) for k in range(uff)])
    #r_DW = np.array([CT_m_res.addVar() for i in range(2*n) for k in range(uff)])
    #R_UP = np.array([CT_m_res.addVar() for i in range(2*n) for k in range(uff)])
    R_UP = np.array([CT_m_res.addVar() for i in range(n) for k in range(uff)])
    #R_DW = np.array([CT_m_res.addVar() for i in range(2*n) for k in range(uff)])
    q_imp = np.array([CT_m_res.addVar() for k in range(uff)])
    q_exp = np.array([CT_m_res.addVar() for k in range(uff)])
    alfa = np.array([CT_m_res.addVar() for i in range(n) for k in range(uff)])
    beta = np.array([CT_m_res.addVar() for i in range(n) for k in range(uff)])
    q_pos = np.array([CT_m_res.addVar() for i in range(n) for k in range(uff)])
    q = np.array([CT_m_res.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(uff)]) 
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    CT_m_res.update()
    
    p = np.transpose(p.reshape(n,uff))
    l = np.transpose(l.reshape(n,uff))
    q = np.transpose(q.reshape(n,uff))
    q_pos = np.transpose(q_pos.reshape(n,uff))
    alfa = np.transpose(alfa.reshape(n,uff))
    beta = np.transpose(beta.reshape(n,uff))
    #r_UP = np.transpose(r_UP.reshape(2*n,uff)) 
    r_UP = np.transpose(r_UP.reshape(n,uff)) 
    #r_DW = np.transpose(r_DW.reshape(2*n,uff))
    #R_UP = np.transpose(R_UP.reshape(2*n,uff))
    R_UP = np.transpose(R_UP.reshape(n,uff))
    #R_DW = np.transpose(R_DW.reshape(2*n,uff))
    
    # Set objective: 
    obj = sum(sum(y0_c[temp,:]*(-l) + mm_c[temp,:]/2*(-l)*(-l)) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) +\
           sum(gamma_i*q_imp + gamma_e*q_exp) + sum(0.001*sum(q_pos)) +\
           sum(sum(CR_UP[temp,:]*R_UP)) +\
           pi_res_UP*(sum(sum(y0_g[temp,:]*(r_UP) + mm_g[temp,:]/2*(r_UP)*(r_UP))))
    CT_m_res.setObjective(obj)      
#sum(sum(CR_DW[temp,:]*R_DW)) +\
#pi_res_UP*(sum(y0_c[temp,:]*(r_UP[:,n:]) + mm_c[temp,:]/2*(r_UP[:,n:])*(r_UP[:,n:])) + sum(y0_g[temp,:]*r_UP[:,:n] + mm_g[temp,:]/2*r_UP[:,:n]*r_UP[:,:n])) +\
#pi_res_DW*(sum(y0_c[temp,:]*(-r_DW[:,n:]) + mm_c[temp,:]/2*(-r_DW[:,n:])*(-r_DW[:,n:])) + sum(y0_g[temp,:]*(-r_DW[:,:n]) + mm_g[temp,:]/2*(-r_DW[:,:n])*(-r_DW[:,:n]))))
    
    
    
    # Add constraint
    for k in range(uff):
        for i in range(n):
            CT_m_res.addConstr(p[k,i] - l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT_m_res.addConstr(q[k,i] <= q_pos[k,i])
            CT_m_res.addConstr(q[k,i] >= -q_pos[k,i])
            CT_m_res.addConstr(p[k,i] - PV[t+k,i] + R_UP[k,i] <= Pmax[i], name="R_UP_limit_p[%s,%s]"% (k,i))
            CT_m_res.addConstr(p[k,i] - PV[t+k,i] >= Pmin[i], name="R_DW_limit_p[%s,%s]"% (k,i))
            #CT_m_res.addConstr(p[k,i] - PV[t+k,i] - R_DW[k,i] >= Pmin[i], name="R_DW_limit_p[%s,%s]"% (k,i))
            #CT_m_res.addConstr(-l[k,i] + R_UP[k,i+n] <= Lmax[t+k,i], name="R_UP_limit_l[%s,%s]"% (k,i))
            CT_m_res.addConstr(-l[k,i] <= Lmax[t+k,i], name="Lmax_l[%s,%s]"% (k,i))
            CT_m_res.addConstr(-l[k,i] >= Lmin[t+k,i], name="Lmin_l[%s,%s]"% (k,i))
            #CT_m_res.addConstr(-l[k,i] - R_DW[k,i+n] >= Lmin[t+k,i], name="R_DW_limit_l[%s,%s]"% (k,i))
        #for i in range(2*n):
            CT_m_res.addConstr(r_UP[k,i] - R_UP[k,i] == 0, name="r_UP_limit[%s,%s]"% (k,i))
            #CT_m_res.addConstr(r_DW[k,i] - R_DW[k,i]<= 0, name="r_DW_limit[%s,%s]"% (k,i))
        CT_m_res.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k))
        CT_m_res.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT_m_res.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
        CT_m_res.addConstr(sum(r_UP[k,:]) - RES_UP[t+k] == 0, name="r_UP_requirement[%s]"%(k))
        #CT_m_res.addConstr(sum(r_DW[k,:]) - RES_DW[t+k] == 0, name="r_DW_requirement[%s]"%(k))
        
    for i in range(n):         
        CT_m_res.addConstr(sum(Agg_load[temp,i] - l[:,i]) == 0)
    CT_m_res.update()    
        
    CT_m_res.optimize()
    for k in range(uff):
        for i in range(n):
            CT_price_sol_res[i,t+k] = CT_m_res.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT_p_sol_res[i,t+k] = p[k,i].x
            CT_l_sol_res[i,t+k] = l[k,i].x
            CT_q_sol_res[i,t+k] = q[k,i].x
            CT_alfa_sol_res[i,t+k] = alfa[k,i].x
            CT_beta_sol_res[i,t+k] = beta[k,i].x
        #for i in range(2*n):
            CT_r_UP_sol_res[i,t+k] = r_UP[k,i].x
            #CT_r_DW_sol_res[i,t+k] = r_DW[k,i].x
            CT_R_UP_sol_res[i,t+k] = R_UP[k,i].x
            #CT_R_DW_sol_res[i,t+k] = R_DW[k,i].x
        CT_price2_sol_res[0,t+k] = CT_m_res.getConstrByName("comm[%s]"%k).Pi
        CT_price2_sol_res[1,t+k] = CT_m_res.getConstrByName("imp_bal[%s]"%k).Pi
        CT_price2_sol_res[2,t+k] = CT_m_res.getConstrByName("exp_bal[%s]"%k).Pi
        CT_imp_sol_res[t+k] = q_imp[k].x
        CT_exp_sol_res[t+k] = q_exp[k].x

CT_IE_sol_res = CT_imp_sol_res - CT_exp_sol_res

cost_res_tp = np.zeros([TMST_run,n]) 
for t in range(TMST_run):
    for p in range(n):
        cost_res_tp[t,p] = y0_c[t,p]*CT_l_sol_res[p,t] + mm_c[t,p]/2*CT_l_sol_res[p,t]*CT_l_sol_res[p,t] + y0_g[t,p]*CT_p_sol_res[p,t] + mm_g[t,p]/2*CT_p_sol_res[p,t]*CT_p_sol_res[p,t]+\
                          CT_alfa_sol_res[p,t]*CT_imp_sol_res[t] + CT_beta_sol_res[p,t]*CT_exp_sol_res[t] + (-CT_price2_sol_res[0,t])*(CT_q_sol_res[p,t]) #-\
                          #CR_UP[t,p]*CT_R_UP_sol_res[p,t]- CR_UP[t,p+n]*CT_R_UP_sol_res[p+n,t] - CR_DW[t,p]*CT_R_DW_sol_res[p,t]- CR_DW[t,p+n]*CT_R_DW_sol_res[p+n,t]

cost_res_tp_times4 = np.repeat(cost_res_tp, 4, axis=0)
cost_res_p = np.sum(cost_res_tp_times4, axis=0)
cost_res = np.sum(cost_res_p)

