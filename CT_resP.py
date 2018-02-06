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
from TMST_def import TMST, TMST_run, n_days

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
     
# balancing prices (every hour)
el_price_DA = el_price_e
el_price_DW = el_price_sampled[:,0]
el_price_UP = el_price_sampled[:,2]


#%% Community CENTRALIZED solution with reserve POP-OUT

# prices
#delta_lambda_UP=np.arange(0,0.02,0.001)
#delta_lambda_DW=-np.arange(0,0.02,0.001)
#delta_lambda_UP = np.average(el_price_UP) - np.average(el_price_e)
#delta_lambda_DW = np.average(el_price_DW) - np.average(el_price_e)
W_resP = np.empty(10)
cost_resP = np.empty(10)
R_UP_avrg = np.empty(10)
R_UP_stdv = np.empty(10)
R_DW_avrg = np.empty(10)
R_DW_stdv = np.empty(10)

#for delta_lambda_UP in np.arange(0,0.02,0.002):
for delta_lambda_UP in range(1):    
    index = int(delta_lambda_UP*1000/2)
    lambda_UP = (el_price_e + 0.1) + delta_lambda_UP*np.ones(len(el_price_e))
    lambda_DW = (el_price_e) -(delta_lambda_UP)*np.ones(len(el_price_e))

# allocate variables
    CT_p_sol_resP = np.zeros((n,TMST_run))
    CT_l_sol_resP = np.zeros((n,TMST_run))
    CT_q_sol_resP = np.zeros((n,TMST_run))
    CT_R_UP_sol_resP = np.zeros(TMST_run)
    CT_R_DW_sol_resP = np.zeros(TMST_run)
    CT_r_p_UP_sol_resP = np.zeros((n,TMST_run))
    CT_r_p_DW_sol_resP = np.zeros((n,TMST_run))
    CT_r_l_UP_sol_resP = np.zeros((n,TMST_run))
    CT_r_l_DW_sol_resP = np.zeros((n,TMST_run))
    CT_alfa_sol_resP = np.zeros((n,TMST_run))
    CT_beta_sol_resP = np.zeros((n,TMST_run))
    CT_imp_sol_resP = np.zeros(TMST_run)
    CT_exp_sol_resP = np.zeros(TMST_run)
    CT_obj_sol_resP = np.zeros(TMST_run)
    CT_price_sol_resP = np.zeros((n,TMST_run))
    CT_price2_sol_resP = np.zeros((3,TMST_run))
    
    uff = np.int32(24*n_days)
    for t in np.arange(0,TMST_run,uff):  # for t = 0, 24
        temp = range(t,t+uff)
        # Create a new model
        CT_m_resP = gb.Model("qp")
        #CT_m.setParam( 'OutputFlag', False ) # Quieting Gurobi output
        # Create variables
        p = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        l = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        r_p_UP = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        r_p_DW = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        r_l_UP = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        r_l_DW = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        R_UP = np.array([CT_m_resP.addVar() for k in range(uff)])
        R_DW = np.array([CT_m_resP.addVar() for k in range(uff)])
        q_imp = np.array([CT_m_resP.addVar() for k in range(uff)])
        q_exp = np.array([CT_m_resP.addVar() for k in range(uff)])
        alfa = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        beta = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        q_pos = np.array([CT_m_resP.addVar() for i in range(n) for k in range(uff)])
        q = np.array([CT_m_resP.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(uff)]) 
        #    gamma_res_i = lambda_UP[temp] # if uncertainty in the bal prices
        #    gamma_res_e = -lambda_DW[temp]
        gamma_res_i = el_price_UP[temp]   # perfect knowledge of balancing prices
        gamma_res_e = -el_price_DW[temp]
        gamma_i = (el_price_e[temp] + 0.1)
        gamma_e = -el_price_e[temp]
        CT_m_resP.update()
        
        p = np.transpose(p.reshape(n,uff))
        l = np.transpose(l.reshape(n,uff))
        q = np.transpose(q.reshape(n,uff))
        r_p_UP = np.transpose(r_p_UP.reshape(n,uff))
        r_p_DW = np.transpose(r_p_DW.reshape(n,uff))
        r_l_UP = np.transpose(r_l_UP.reshape(n,uff))
        r_l_DW = np.transpose(r_l_DW.reshape(n,uff))
        q_pos = np.transpose(q_pos.reshape(n,uff))
        alfa = np.transpose(alfa.reshape(n,uff))
        beta = np.transpose(beta.reshape(n,uff))
        
        
        # Set objective: 
        obj = (sum(sum(y0_c[temp,:]*(-l+r_l_UP-r_l_DW) + mm_c[temp,:]/2*(-l+r_l_UP-r_l_DW)*(-l+r_l_UP-r_l_DW)) +\
                       sum(y0_g[temp,:]*(p+r_p_UP-r_p_DW) + mm_g[temp,:]/2*(p+r_p_UP-r_p_DW)*(p+r_p_UP-r_p_DW)) +\
                       sum(gamma_i*q_imp + gamma_e*q_exp) +\
                       sum(0.001*sum(q_pos)) +\
                       -sum(gamma_res_i*R_UP + gamma_res_e*R_DW)))
        CT_m_resP.setObjective(obj)
            
            # Add constraint
        for k in range(uff):
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
        for k in range(uff):
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
                
    W_resP_tp = np.zeros([TMST_run,n]) 
    for t in range(TMST_run):
        for p in range(n):
            W_resP_tp[t,p] = CT_alfa_sol_resP[p,t]*(el_price_e[t]+0.1) + CT_beta_sol_resP[p,t]*(-el_price_e[t]) +\
            y0_c[t,p]*(-CT_l_sol_resP[p,t]) + mm_c[t,p]/2*(-CT_l_sol_resP[p,t])*(-CT_l_sol_resP[p,t]) + y0_g[t,p]*CT_p_sol_resP[p,t] + mm_g[t,p]/2*CT_p_sol_resP[p,t]*CT_p_sol_resP[p,t]+\
                            + (-CT_price2_sol_resP[0,t])*(CT_q_sol_resP[p,t])
                            #CT_alfa_sol_resP[p,t]*(-CT_price2_sol_resP)[1,t] + CT_beta_sol_resP[p,t]*(-CT_price2_sol_resP)[2,t] + (-CT_price2_sol_resP[0,t])*(CT_q_sol_resP[p,t])
    
    W_resP_p = np.sum(W_resP_tp, axis=0)
    
    cost_resP_tp = np.zeros([TMST_run,n]) 
    for t in range(TMST_run):
        for p in range(n):
            cost_resP_tp[t,p] = (-CT_price2_sol_resP[0,t])*(CT_q_sol_resP[p,t]) + CT_alfa_sol_resP[p,t]*(el_price_e[t]+0.1) + CT_beta_sol_resP[p,t]*(-el_price_e[t])
        
#cost_resP_tp_times4 = np.repeat(cost_resP_tp, 4, axis=0)
    cost_resP_p = np.sum(cost_resP_tp, axis=0)
    cost_resP = np.sum(cost_resP_p)
#    # saving results in these arrays
#    W_resP[index] = np.sum(W_resP_p)
#    cost_resP[index] = np.sum(cost_resP_p)
#    R_UP_avrg[index] = np.average(CT_R_UP_sol_resP)
#    R_UP_stdv[index] = np.std(CT_R_UP_sol_resP)
#    R_DW_avrg[index] = np.average(CT_R_DW_sol_resP)
#    R_DW_stdv[index] = np.std(CT_R_DW_sol_resP)
    