# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:40:18 2016

@author: fmoret
"""
import os
import gurobipy as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shelve


dd1 = os.getcwd() #os.path.realpath(__file__) #os.getcwd()
data_path = str(Path(dd1).parent.parent)+r'\trunk\Input Data 2'

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
TMST = 48#b2.shape[0]

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
CT_p_sol = np.zeros((g,TMST))
CT_l_sol = np.zeros((n,TMST))
CT_q_sol = np.zeros((n,TMST))
CT_alfa_sol = np.zeros((n,TMST))
CT_beta_sol = np.zeros((n,TMST))
CT_imp_sol = np.zeros(TMST)
CT_exp_sol = np.zeros(TMST)
CT_obj_sol = np.zeros(TMST)
CT_price_sol = np.zeros((n,TMST))
CT_price2_sol = np.zeros((3,TMST))

for t in np.arange(0,TMST,24):  # for t = 0, 24
    temp = range(t,t+24)
    # Create a new model
    CT_m = gb.Model("qp")
    CT_m.setParam( 'OutputFlag', False ) # Quieting Gurobi output
    # Create variables
    p = np.array([CT_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
    l = np.array([CT_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
    q_imp = np.array([CT_m.addVar() for k in range(24)])
    q_exp = np.array([CT_m.addVar() for k in range(24)])
    alfa = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    beta = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    q_pos = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    q = np.array([CT_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)]) 
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    CT_m.update()
    
    p = np.transpose(p.reshape(n,24))
    l = np.transpose(l.reshape(n,24))
    q = np.transpose(q.reshape(n,24))
    q_pos = np.transpose(q_pos.reshape(n,24))
    alfa = np.transpose(alfa.reshape(n,24))
    beta = np.transpose(beta.reshape(n,24))
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(0.001*sum(q_pos)))
    # doppia sommatoria: 
    # la somma interna è sulle colonne, quindi da 15x24 si passa a 24
    # quindi somma sui prosumer
    # la somma esterna somma tutti gli elementi dell'array
    # quindi somma sulle 24 ore --> un giorno
    # discorso simile per le altre sommatorie
    # l'objective è minimizzare il totale di un giorno

    CT_m.setObjective(obj)
    
    # Add constraint
    for k in range(24):
        CT_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k)) #somma sui prosumer
        #what about mettere una variabile invece che 0? da minimizzare poi nell'obj
        for i in range(n): #balance per ogni prosumer, ogni ora
            CT_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT_m.addConstr(q[k,i] <= q_pos[k,i]) #limite sulla q, che può essere pos o neg
            CT_m.addConstr(q[k,i] >= -q_pos[k,i])
        CT_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
    for i in range(n): #nell'arco di tutto il giorno i conti sul carico devono tornare 
    #per ogni prosumer, ma l'ottimizzazione è fatta ora per ora
        CT_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
#        for j in np.arange(0,24,window):
#            CT_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
    CT_m.update()    
        
    CT_m.optimize()
    for k in range(24):
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

##crea immagini
#for k in range(3):
#    prosumers = ['pros%k']
#    style = ['r','g','y',]
#    plt.plot(range(48), CT_p_sol[k,:] )
#    legend()
#
#d={}
#a=np.empty(9)
#for x in range(1,10):
#       a[x]=d["string{0}".format(x)]


#m = err
limit_comply_consumption = (CT_l_sol.T<=Lmax[:48,:])&(CT_l_sol.T>=Lmin[:48,:])
limit_comply_production = np.zeros([48,15])
for p in range(15):
    for t in range(48):
        limit_comply_production[t,p] = (CT_p_sol.T[t,p]-PV[t,p]<=Pmax[p])&(CT_p_sol.T[t,p]-PV[t,p]>=Pmin[p])

#the next one does'nt work
price_comply = np.zeros([48,15])
case = np.zeros([48,15])
for p in range(15):
    for t in range(48):
        if (CT_price2_sol[0,t]>(b2[t,p] + c2[t,p]-0.01)):
            price_comply[t,p] = (CT_l_sol.T[t,p]==Lmax[t,p])
            case[t,p] = 1
        elif (CT_price2_sol[0,t]<(b2[t,p] - c2[t,p]-0.01)):
            price_comply[t,p] = (CT_l_sol.T[t,p]==Lmin[t,p])
            case[t,p] = 2
        else:
            price_comply[t,p] = (CT_l_sol.T[t,p]>=Lmin[t,p])&(CT_l_sol.T[t,p]<=Lmax[t,p])
            case[t,p] = 3



#%% ADMM optimisation CED via master-sub classes
from ADMM_master import ADMM_Master
o = 0.01
e = 0.00001
asynch = -100
weight = 1/n

#% Cost of Import/Export not considered ==> case = 0
Mast = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch)
ADMM_summary = Mast.results
ADMM_prices = Mast.prices
ADMM_flag = Mast.flag


#%%
max_l = abs(Mast.variables.l_k - np.transpose(CT_l_sol)).max()
max_p = abs(Mast.variables.p_k - np.transpose(CT_p_sol)).max()
max_q = abs(Mast.variables.q_k - np.transpose(CT_q_sol)).max()
max_alfa = abs(Mast.variables.alfa_k - np.transpose(CT_alfa_sol)).max()
max_beta = abs(Mast.variables.beta_k - np.transpose(CT_beta_sol)).max()
l_1 = Mast.variables.l_k 
p_1 = Mast.variables.p_k
q_1 = Mast.variables.q_k
q_pos_1 = Mast.variables.q_pos_k
alfa_1 = Mast.variables.alfa_k 
beta_1 = Mast.variables.beta_k 
imp_1 = alfa_1.sum(axis=1) 
exp_1 = beta_1.sum(axis=1)
ie_1 = imp_1-exp_1
price_community = Mast.variables.price_comm


opt1 = np.empty(TMST)
ooo = np.empty(TMST)
for i1 in range(TMST):
    opt1[i1] = (sum(y0_c[i1,:]*l_1[i1,:] + mm_c[i1,:]/2*l_1[i1,:]*l_1[i1,:]) + sum(y0_g[i1,:]*p_1[i1,:] + mm_g[i1,:]/2*p_1[i1,:]*p_1[i1,:]) #+ el_price_e[i1]*ie_1[i1]
                + 0.001*sum(abs(q_1[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_1[i1,:]) - el_price_e[i1]*sum(beta_1[i1,:]))
    ooo[i1] = (sum(y0_c[i1,:]*CT_l_sol[:,i1] + mm_c[i1,:]/2*CT_l_sol[:,i1]*CT_l_sol[:,i1]) + sum(y0_g[i1,:]*CT_p_sol[:,i1] + mm_g[i1,:]/2*CT_p_sol[:,i1]*CT_p_sol[:,i1]) + 
                (el_price_e[i1]+0.1)*CT_imp_sol[i1] - el_price_e[i1]*CT_exp_sol[i1] + 0.001*sum(abs(CT_q_sol[:,i1]))) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
    
gap_opt1 = opt1-CT_obj_sol
gap_opt12 = opt1-ooo

#%% BAL - definition of parameters and utility and cost curves
#os.chdir(dd2)
#import shelve
#filename='input_data_2.out'

# input data
d = shelve.open(filename, 'r')
el_price_dataframe = pd.read_excel(file_loc)/1000
el_price_notSampled = el_price_dataframe.values
el_price_sampled = el_price_notSampled[1::2] #ogni 2 valori prende il secondo valore
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

TMST=48*4
# data extension 1h --> 15 mins (repeat each row 4 times)
b2 = np.repeat(b2, 4, axis=0)
c2 = np.repeat(c2, 4, axis=0)
b1 = np.repeat(b1, 4, axis=0)
c1 = np.repeat(c1, 4, axis=0)
PV = np.repeat(PV, 4, axis=0)
Load = np.repeat(Load, 4, axis=0)
Flex_load = np.repeat(Flex_load, 4, axis=0)


# create noise for PV (only if PV nonzero) 
noise_PV = np.random.normal(0,0.1,(PV.shape))
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
noise_Load = np.random.normal(0,0.1,(Load.shape))

# adding noise to Load (and if the result is less than zero bring it to zero)
Load_real = Load + noise_Load
for row in range(Load_real.shape[0]):
    for col in range(Load_real.shape[1]):
        if Load_real[row,col] < 0:
            Load_real[row,col] = 0
            
# IMBALANCES
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
##EXPECTED imbalance
#deltaPV_exp = deltaPV.sum()/(n*TMST)
#deltaLoad_exp = deltaLoad.sum()/(n*TMST)


# aggregated load before and after imbalance            
Agg_load = Load + Flex_load
Agg_load_real = Load_real + Flex_load

# other stuff - including: How many days to RUN
PostCode = d['PostCode']
os.chdir(dd1)
n = b2.shape[1]
g = b1.shape[1]
TMST = 48*4#b2.shape[0] # *4 because now we have 15 mins time intervals

# retrieving and extending (1h --> 15min) set points and DA price from DA stage
p_tilde = np.repeat(p_1, 4, axis=0)
#p_tilde = np.repeat(CT_p_sol.T, 4, axis=0)
l_tilde = np.repeat(l_1, 4, axis=0)
#l_tilde = np.repeat(CT_l_sol.T, 4, axis=0)
lambda_CED = np.repeat(price_community, 4, axis=0)
#lambda_CED = np.repeat(-CT_price2_sol[0,:], 4, axis=0)

# save old Pmax, Pmin, Lmax, Lmin, intercepts and slopes and extend them (they'll be referred to 15 mins time interval)
Pmax_DA = Pmax
Pmin_DA = Pmin
Lmax_DA = np.repeat(Lmax, 4, axis=0)
Lmin_DA = np.repeat(Lmin, 4, axis=0)
y0_c_DA = np.repeat(y0_c, 4, axis=0)
y0_g_DA = np.repeat(y0_g, 4, axis=0)
mm_c_DA = np.repeat(mm_c, 4, axis=0)
mm_g_DA = np.repeat(mm_g, 4, axis=0)

# new Pmax and Pmin, Lmax and Lmin (referred to 15 mins time interval)----CHECK if correct as a concept!!!
Pmax_bal = np.zeros((TMST,n))
Pmin_bal = np.zeros((TMST,n))
Lmax_bal = np.zeros((TMST,n))
Lmin_bal = np.zeros((TMST,n))
for t in range(TMST):
    for p in range(n):
        Pmax_bal[t,p] = Pmax_DA[p] + PV[t,p] - p_tilde[t,p] 
        Pmin_bal[t,p] = Pmin_DA[p] + PV[t,p] - p_tilde[t,p]
        Lmax_bal[t,p] = Lmax_DA[t,p] - l_tilde[t,p]
        Lmin_bal[t,p] = Lmin_DA[t,p] - l_tilde[t,p]

    
# TRASLATING the CURVES - using directly the DA price - CHECK if the results are the same as with the other method
y0_c_bal = np.zeros((TMST,n))
y0_g_bal = np.zeros((TMST,n))

#CONSUMERS
# intercept
for t in range(TMST):
    for p in range(n):
        y0_c_bal[t,p] = y0_c_DA[t,p] + mm_c_DA[t,p]*l_tilde[t,p]
# slope
mm_c_bal = mm_c_DA

# GENERATORS        
# intercept  
for t in range(TMST):
    for p in range(n):
        y0_g_bal[t,p] = y0_g_DA[t,p] + mm_g_DA[t,p]*p_tilde[t,p]  
# slope
mm_g_bal = mm_g_DA


# OTHER METHOD
## consumers: the intercept is now the DA price, the slope of the curve changes and it's determined by the new Lmax and Lmin
## intercept
#for t in range(TMST):
#    for p in range(n):
#        if l_tilde[t,p] == Lmin_DA[t,p]:
#            y0_c_bal[t,p] = 0.01 + b2[t,p] - c2[t,p]
#        elif l_tilde[t,p] == Lmax_DA[t,p]:
#            y0_c_bal[t,p] = 0.01 + b2[t,p] + c2[t,p]
#        else:
#            y0_c_bal[t,p] = lambda_CED[t]

## slope (removing the infinite values, rising from dividing by zero)
#mm_c_bal = 2*c2[:TMST,:]/(Lmax_bal-Lmin_bal)
#if sum(abs(mm_c_bal[np.isinf(mm_c_bal)]))>0:
#        y0_c_bal[np.isinf(abs(mm_c_bal))] = 0.01 + b2[np.isinf(abs(mm_c_bal))]
#        mm_c_bal[np.isinf(abs(mm_c_bal))] = 0

## generators: the intercept now is the DA price, the slope of the curve does not change and it's determined by Pmax and Pmin
#for t in range(TMST):
## intercept
#    for p in range(n):
#        if p_tilde[t,p] - PV[t,p] == Pmin_DA[p]:
#            y0_g_bal[t,p] = 0.01 + b1[t,p] - c1[t,p]
#        elif p_tilde[t,p] - PV[t,p] == Pmax_DA[p]:
#            y0_g_bal[t,p] = 0.01 + b1[t,p] + c1[t,p]
#        else:
#            y0_g_bal[t,p] = lambda_CED[t]
#            
## slope (removing the infinite values and the NaN values)
#mm_g_bal = 2*c1[:TMST,:]/(Pmax_bal-Pmin_bal)
#if sum(abs(mm_g_bal[np.isinf(mm_g_bal)]))>0:
#        y0_g_bal[np.isinf(abs(mm_g_bal))] = 0.01 + b1[np.isinf(abs(mm_g_bal))]
#        mm_g_bal[np.isinf(abs(mm_g_bal))] = 0
#        y0_g_bal[np.isnan(y0_g_bal)] = 0
#        mm_g_bal[np.isnan(mm_g_bal)] = 0

## TRASLATING the CURVES - creating them again based on new Lmin and Lmax
## consumers
#y0_c_bal = 0.01 + b2[:TMST,:] + (l_tilde-(Lmax-Lmin)/2)*y0_c_DA[:TMST,:] # controlla che sia positivo
## generators
#y0_g_bal = 0.01 + b1[:TMST,:] + (p_tilde-(Pmax-Pmin)/2)*(2*c2[:TMST,:]/(Pmax-Pmin)) # controlla che sia positivo


tau = 0.1
window = 4

#%% SOME PICTURES

#PV imbalance - prosumer 0
fig_PV = plt.figure(11,figsize=[10,6])
ax1, = plt.plot(PV[:192,0],label='PV forecast')
ax2, = plt.plot(PV_real[:192,0],label='PV realization')
plt.legend([ax1, ax2], ['PV forecast', 'PV realization'])
plt.xlabel('time [15 mins intervals]')
plt.ylabel('PV production [kWh]')
plt.title('PV production prosumer #0')
#load imbalance - prosumer 1
fig_load = plt.figure(11,figsize=[10,6])
ax1, = plt.plot(Load[:192,1],label='Load forecast')
ax2, = plt.plot(Load_real[:192,1],label='Load realization')
plt.legend([ax1, ax2], ['Load forecast', 'Load realization'])
plt.xlabel('time [15 mins intervals]')
plt.ylabel('Load comsumption [kWh]')
plt.title('Load consumption prosumer #1')
#load imbalance  and minmax limits - prosumer 1
fig_load_minmax = plt.figure(11,figsize=[10,6])
ax1, = plt.plot(-Load[:192,1],label='Load forecast',marker='o',linestyle='dashed')
ax2, = plt.plot(-Load_real[:192,1],label='Load realization',linestyle='dashed')
ax3, = plt.plot(Lmin_DA[:192,1],label='Lmin')
ax4, = plt.plot(Lmax_DA[:192,1],label='Lmax')
ax5, = plt.plot(l_tilde[:192,1],label='consumption set point',linestyle='dashed')
plt.legend([ax1, ax2,ax3,ax4,ax5], ['Load forecast', 'Load realization','Lmin','Lmax','consumption set point'])
plt.xlabel('time [15 mins intervals]')
plt.ylabel('Load comsumption [kWh]')
plt.title('Load consumption prosumer #1')

#%% Community CENTRALIZED solution with reserve

RES_UP = imbalance_community        # reserve requirement - set depending on the average imbalance of the community each hour
RES_DW = RES_UP
CR_UP = 0.001*np.ones([TMST,2*n])
CR_DW = 0.001*np.ones([TMST,2*n])
pi_res_UP = 0.05
pi_res_DW = 0.05  


CT_p_sol = np.zeros((g,TMST))
CT_l_sol = np.zeros((n,TMST))
CT_q_sol = np.zeros((n,TMST))
CT_r_UP_sol = np.zeros((n,TMST))
CT_r_DW_sol = np.zeros((n,TMST))
CT_R_UP_sol = np.zeros((n,TMST))
CT_R_DW_sol = np.zeros((n,TMST))
CT_alfa_sol = np.zeros((n,TMST))
CT_beta_sol = np.zeros((n,TMST))
CT_imp_sol = np.zeros(TMST)
CT_exp_sol = np.zeros(TMST)
CT_obj_sol = np.zeros(TMST)
CT_price_sol = np.zeros((n,TMST))
CT_price2_sol = np.zeros((3,TMST))

for t in np.arange(0,TMST,24):  # for t = 0, 24
    temp = range(t,t+24)
    # Create a new model
    CT_m = gb.Model("qp")
    #CT_m.setParam( 'OutputFlag', False ) # Quieting Gurobi output
    # Create variables
    p = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    l = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    r_UP = np.array([CT_m.addVar() for i in range(2*n) for k in range(24)])
    r_DW = np.array([CT_m.addVar(lb=-gb.GRB.INFINITY, ub=0) for i in range(2*n) for k in range(24)])
    R_UP = np.array([CT_m.addVar() for i in range(2*n) for k in range(24)])
    R_DW = np.array([CT_m.addVar() for i in range(2*n) for k in range(24)])
    q_imp = np.array([CT_m.addVar() for k in range(24)])
    q_exp = np.array([CT_m.addVar() for k in range(24)])
    alfa = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    beta = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    q_pos = np.array([CT_m.addVar() for i in range(n) for k in range(24)])
    q = np.array([CT_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)]) 
    gamma_i = (el_price_e[temp] + 0.1)
    gamma_e = -el_price_e[temp]
    CT_m.update()
    
    p = np.transpose(p.reshape(n,24))
    l = np.transpose(l.reshape(n,24))
    q = np.transpose(q.reshape(n,24))
    q_pos = np.transpose(q_pos.reshape(n,24))
    alfa = np.transpose(alfa.reshape(n,24))
    beta = np.transpose(beta.reshape(n,24))
    r_UP = np.transpose(r_UP.reshape(2*n,24)) 
    r_DW = np.transpose(r_DW.reshape(2*n,24))
    R_UP = np.transpose(R_UP.reshape(2*n,24))
    R_DW = np.transpose(R_DW.reshape(2*n,24))
    
    
    # Set objective: 
    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) +\
           sum(gamma_i*q_imp + gamma_e*q_exp) + sum(0.001*sum(q_pos))) +\
           sum(sum(CR_UP[temp,:]*R_UP) + sum(CR_DW[temp,:]*R_DW)) +\
           pi_res_UP*(sum(sum(y0_c[temp,:]*r_UP[:,:n] + mm_c[temp,:]/2*r_UP[:,:n]*r_UP[:,:n]) + sum(-y0_g[temp,:]*r_UP[:,n:] - mm_g[temp,:]/2*r_UP[:,n:]*r_UP[:,n:]))) +\
           pi_res_DW*(sum(sum(y0_c[temp,:]*r_DW[:,:n] + mm_c[temp,:]/2*r_DW[:,:n]*r_DW[:,:n]) + sum(-y0_g[temp,:]*r_DW[:,n:] - mm_g[temp,:]/2*r_DW[:,n:]*r_DW[:,n:])))
    CT_m.setObjective(obj)
    
    # Add constraint
    for k in range(24):
        for i in range(n):
            CT_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT_m.addConstr(q[k,i] <= q_pos[k,i])
            CT_m.addConstr(q[k,i] >= -q_pos[k,i])
            CT_m.addConstr(p[k,i] - PV[k,i] + R_UP[k,i] <= Pmax[i], name="R_UP_limit_p[%s,%s]"% (k,i))
            CT_m.addConstr(p[k,i] - PV[k,i] - R_DW[k,i] >= Pmin[i], name="R_DW_limit_p[%s,%s]"% (k,i))
            CT_m.addConstr(l[k,i] + R_UP[k,i+n] <= Lmax[k,i], name="R_UP_limit_l[%s,%s]"% (k,i))
            CT_m.addConstr(l[k,i] - R_DW[k,i+n] >= Lmin[k,i], name="R_DW_limit_l[%s,%s]"% (k,i))
        for i in range(2*n):
            CT_m.addConstr(r_UP[k,i] <= R_UP[k,i], name="r_UP_limit[%s,%s]"% (k,i))
            CT_m.addConstr(-r_DW[k,i] <= R_DW[k,i], name="r_DW_limit[%s,%s]"% (k,i))
        CT_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k))
        CT_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
        CT_m.addConstr(sum(r_UP[k,:]) - RES_UP[k] == 0, name="r_UP_requirement[%s]"%(k))
        CT_m.addConstr(-sum(r_DW[k,:]) - RES_DW[k] == 0, name="r_DW_requirement[%s]"%(k))
        
    for i in range(n): #nell'arco di tutto il giorno i conti sul carico devono tornare 
    #per ogni prosumer, ma l'ottimizzazione è fatta ora per ora
        CT_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
#        for j in np.arange(0,24,window):
#            CT_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
    CT_m.update()    
        
    CT_m.optimize()
    for k in range(24):
        for i in range(n):
            CT_price_sol[i,t+k] = CT_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
            CT_p_sol[i,t+k] = p[k,i].x
            CT_l_sol[i,t+k] = l[k,i].x
            CT_q_sol[i,t+k] = q[k,i].x
            CT_alfa_sol[i,t+k] = alfa[k,i].x
            CT_beta_sol[i,t+k] = beta[k,i].x
        for i in range(2*n):
            CT_r_UP_sol[i,t+k] = r_UP[k,i].x
            CT_r_DW_sol[i,t+k] = r_DW[k,i].x
            CT_R_UP_sol[i,t+k] = R_UP[k,i].x
            CT_R_DW_sol[i,t+k] = R_DW[k,i].x
        CT_price2_sol[0,t+k] = CT_m.getConstrByName("comm[%s]"%k).Pi
        CT_price2_sol[1,t+k] = CT_m.getConstrByName("imp_bal[%s]"%k).Pi
        CT_price2_sol[2,t+k] = CT_m.getConstrByName("exp_bal[%s]"%k).Pi
        CT_imp_sol[t+k] = q_imp[k].x
        CT_exp_sol[t+k] = q_exp[k].x
# http://www.gurobi.com/documentation/7.5/refman/attributes.html
    del CT_m

CT_IE_sol = CT_imp_sol - CT_exp_sol
            

#%% SCENARIOS

# prices
el_price_DA = np.repeat(el_price_e, 4, axis=0)
el_price_DW = np.repeat(el_price_sampled[:,0], 4, axis=0) # 2 rows
el_price_UP = np.repeat(el_price_sampled[:,2], 4, axis=0) # 2 rows
#el_price_UP = el_price_DA + abs(np.random.normal(0,0.5*average_DA_price,(el_price_DA.shape)))
#el_price_DW = el_price_DA - abs(np.random.normal(0,0.5*average_DA_price,(el_price_DA.shape)))

system_state = np.empty([TMST])
for t in range(TMST):
    if el_price_DA[t] == el_price_DW[t]:
        system_state[t] = 1
    elif el_price_DA[t] == el_price_UP[t]:
        system_state[t] = 2
    elif el_price_DW[t] == el_price_UP[t]:
        system_state = 0
#system_state = np.random.randint(3, size=len(el_price_DA)) # where 0 is balanced, 1 is UP and 2 is DW regulation.

el_price_BAL = np.empty([TMST]) # 1 row 
for t in range(TMST):
    if system_state[t] == 0:                # where 0 is balanced, 1 is UP and 2 is DW regulation
        el_price_BAL[t] = el_price_UP[t]
    elif system_state[t] == 1:
        el_price_BAL[t] = el_price_UP[t]
    elif system_state[t] == 2:
        el_price_BAL[t] = el_price_DW[t]
    
ret_price_exp = np.average(el_price_BAL)
ret_price_imp = ret_price_exp + 0.1
el_price_BAL_fixed = np.average(el_price_BAL)*np.ones(TMST)

# benchmark - BRP
# scenario 1 - fixed tariff
imbal_cost_1 = np.zeros([TMST,n])
for t in range(TMST):
    for p in range(n):
        if (deltaPV[t,p]-deltaLoad[t,p]) < 0:
            imbal_cost_1[t,p] = - ret_price_imp*(deltaPV[t,p]-deltaLoad[t,p])
        else:
            imbal_cost_1[t,p] = -ret_price_exp*(deltaPV[t,p]-deltaLoad[t,p])
imbal_cost_p_1 = np.sum(imbal_cost_1, axis = 0)
imbal_cost_p_perunit_imbal_1 = -abs(imbal_cost_p_1)/imbalance_prosumer
imbal_cost_1 = np.sum(imbal_cost_1)

# scenario 2 - dynamic tariff - 2 price system
imbal_cost_2 = np.zeros([TMST,n])
for t in range(TMST):
    for p in range(n):
        if system_state[t] == 0:
            imbal_cost_2[t,p] = - el_price_DA[t]*(deltaPV[t,p]-deltaLoad[t,p])
        if system_state[t] == 1:
            if (deltaPV[t,p]-deltaLoad[t,p]) < 0:
                imbal_cost_2[t,p] = - el_price_UP[t]*(deltaPV[t,p]-deltaLoad[t,p])
            else:
                imbal_cost_2[t,p] = -el_price_DA[t]*(deltaPV[t,p]-deltaLoad[t,p])
        if system_state[t] == 2:
            if (deltaPV[t,p]-deltaLoad[t,p]) < 0:
                imbal_cost_2[t,p] = - el_price_DA[t]*(deltaPV[t,p]-deltaLoad[t,p])
            else:
                imbal_cost_2[t,p] = -el_price_DW[t]*(deltaPV[t,p]-deltaLoad[t,p])
imbal_cost_p_2 = np.sum(imbal_cost_2, axis = 0)
imbal_cost_p_perunit_imbal_2 = -abs(imbal_cost_p_2)/imbalance_prosumer
imbal_cost_2 = np.sum(imbal_cost_2)


# SCENARI INTERMEDI
imbal_cost_3 = np.zeros([TMST])
for t in range(TMST):
    if imbalance_community[t] < 0:
        imbal_cost_3[t] = - (el_price_BAL_fixed[t] + 0.1)*(imbalance_community[t])
    else:
        imbal_cost_3[t] = - el_price_BAL_fixed[t]*(imbalance_community[t])
imbal_cost_3 = np.sum(imbal_cost_3)
imbal_weights = abs(imbalance_prosumer)/(abs(imbalance_prosumer).sum())
imbal_cost_p_3 = np.empty(n)
for p in range(n):
    imbal_cost_p_3[p] = imbal_weights[p]*imbal_cost_3
imbal_cost_p_perunit_imbal_3 = imbal_cost_p_3/abs(imbalance_prosumer)

imbal_cost_4 = np.zeros([TMST])
for t in range(TMST):
    if system_state[t] == 0:
            imbal_cost_4[t,p] = - el_price_DA[t]*imbalance_community[t]
    if system_state[t] == 1:
        if imbalance_community[t] < 0:
            imbal_cost_4[t] = - el_price_UP[t]*imbalance_community[t]
        else:
            imbal_cost_4[t] = - el_price_DA[t]*imbalance_community[t]
    if system_state[t] == 2:
        if imbalance_community[t] < 0:
            imbal_cost_4[t] = - el_price_DA[t]*imbalance_community[t]
        else:
            imbal_cost_4[t] = - el_price_DW[t]*imbalance_community[t]
imbal_cost_4 = np.sum(imbal_cost_4)
imbal_weights = abs(imbalance_prosumer)/(abs(imbalance_prosumer).sum())
imbal_cost_p_4 = np.empty(n)
for p in range(n):
    imbal_cost_p_4[p] = imbal_weights[p]*imbal_cost_4
imbal_cost_p_perunit_imbal_4 = imbal_cost_p_4/abs(imbalance_prosumer)


#%% Community BALANCING - CENTRALIZED solution - SCENARIO 5 and 6
# choose here:
scenario = 5

CT_p_sol_bal = np.zeros((g,TMST))
CT_l_sol_bal = np.zeros((n,TMST))
CT_q_sol_bal = np.zeros((n,TMST))
CT_alfa_sol_bal = np.zeros((n,TMST))
CT_beta_sol_bal = np.zeros((n,TMST))
CT_imp_sol_bal = np.zeros(TMST)
CT_exp_sol_bal = np.zeros(TMST)
CT_obj_sol_bal = np.zeros(TMST)
CT_price_sol_bal = np.zeros((n,TMST))
CT_price2_sol_bal = np.zeros((3,TMST))

for t in np.arange(0,TMST,96):  # for t = 0, 24
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
    if scenario == 5:
        gamma_i = (el_price_BAL_fixed[temp] + 0.1) 
        gamma_e = -el_price_BAL_fixed[temp]     
    if scenario == 6:
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
        CT_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k)) #somma sui prosumer
        for i in range(n): #balance per ogni prosumer, ogni ora
            CT_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] + deltaPV[k,i] - deltaLoad[k,i] == 0, name="pros[%s,%s]"% (k,i))
            CT_m.addConstr(q[k,i] <= q_pos[k,i]) #limite sulla q, che può essere pos o neg
            CT_m.addConstr(q[k,i] >= -q_pos[k,i])
        CT_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
        CT_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
    ####for i in range(n): #nell'arco di tutto il giorno i conti sul carico devono tornare 
    #per ogni prosumer, ma l'ottimizzazione è fatta ora per ora
        ####CT_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
#        for j in np.arange(0,24,window):
#            CT_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
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

##crea immagini
#for k in range(3):
#    prosumers = ['pros%k']
#    style = ['r','g','y',]
#    plt.plot(range(48), CT_p_sol[k,:] )
#    legend()
#
#d={}
#a=np.empty(9)
#for x in range(1,10):
#       a[x]=d["string{0}".format(x)]

# IMBALANCE COSTS
if scenario == 5:
    #total costs
    imbal_cost_5 = np.empty([TMST])
    for i in range(TMST):
        imbal_cost_5[i] = (sum(y0_c_bal[i,:]*CT_l_sol_bal[:,i] + mm_c_bal[i,:]/2*CT_l_sol_bal[:,i]*CT_l_sol_bal[:,i]) + sum(y0_g_bal[i,:]*CT_p_sol_bal[:,i] + mm_g_bal[i,:]/2*CT_p_sol_bal[:,i]*CT_p_sol_bal[:,i]) + 
                    (el_price_BAL_fixed[i]+0.1)*CT_imp_sol_bal[i] - el_price_BAL_fixed[i]*CT_exp_sol_bal[i] + 0.001*sum(abs(CT_q_sol_bal[:,i])))
    imbal_cost_5 = np.sum(imbal_cost_5)
    #cost each prosumer each time interval
    costrev_5 = np.empty([TMST,n])
    numerator_5 = np.empty([TMST,n])
    denominator_5 = np.empty([TMST,n])
    for t in range(TMST):
        for p in range(n):
            costrev_5[t,p] =0.001*abs(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_BAL_fixed[t] + CT_alfa_sol_bal[p,t]*(el_price_BAL_fixed[t]+0.1) + \
                            y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
            numerator_5[p,t] = -0.001*abs(CT_q_sol_bal[p,t]) + CT_beta_sol_bal[p,t]*el_price_BAL_fixed[t] - CT_alfa_sol_bal[p,t]*(el_price_BAL_fixed[t]+0.1)
            denominator_5[p,t] = CT_l_sol_bal[p,t] + CT_p_sol_bal[p,t]
    perceived_price_5 = np.sum(numerator_5, axis = 0)/np.sum(denominator_5, axis = 0)
    sigma_5 = np.std(perceived_price_5)
    sigmaMax_5 = 0.1
    QoE_5 = 1 - sigma_5/sigmaMax_5
    imbal_cost_p_5 = np.sum(costrev_5, axis=0)
    costrev_5_tot = np.sum(imbal_cost_p_5) # just to check that this is equal to imbal_cost_5
    imbal_cost_p_perunit_imbal_5 = -abs(imbal_cost_p_5)/imbalance_prosumer

if scenario == 6:
    #total costs
    imbal_cost_6 = np.empty([TMST])
    for i in range(TMST):
        imbal_cost_6[i] = (sum(y0_c_bal[i,:]*CT_l_sol_bal[:,i] + mm_c_bal[i,:]/2*CT_l_sol_bal[:,i]*CT_l_sol_bal[:,i]) + sum(y0_g_bal[i,:]*CT_p_sol_bal[:,i] + mm_g_bal[i,:]/2*CT_p_sol_bal[:,i]*CT_p_sol_bal[:,i]) + 
                  el_price_UP[i]*CT_imp_sol_bal[i] - el_price_DW[i]*CT_exp_sol_bal[i] + 0.001*sum(abs(CT_q_sol_bal[:,i]))) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
    imbal_cost_6 = np.sum(imbal_cost_6)
    #cost each prosumer each time interval
    costrev_6 = np.empty([TMST,n])
    for t in range(TMST):
        for p in range(n):
            if system_state[t] == 2:
                costrev_6[t,p] = 0.001*abs(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DW[t] + CT_alfa_sol_bal[p,t]*el_price_DA[t] + \
                y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
            elif system_state[t] == 1:
                costrev_6[t,p] = 0.001*abs(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DA[t] + CT_alfa_sol_bal[p,t]*el_price_UP[t] + \
                y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
            else:
                costrev_6[t,p] = 0.001*abs(CT_q_sol_bal[p,t]) - CT_beta_sol_bal[p,t]*el_price_DA[t] + CT_alfa_sol_bal[p,t]*el_price_DA[t] + \
                y0_c_bal[t,p]*CT_l_sol_bal[p,t] + mm_c_bal[t,p]/2*CT_l_sol_bal[p,t]*CT_l_sol_bal[p,t] + y0_g_bal[t,p]*CT_p_sol_bal[p,t] + mm_g_bal[t,p]/2*CT_p_sol_bal[p,t]*CT_p_sol_bal[p,t]
    imbal_cost_p_6= np.sum(costrev_6, axis=0)
    costrev_6_tot = np.sum(imbal_cost_p_6) # just to check that this is equal to imbal_cost_6
    imbal_cost_p_perunit_imbal_6 = -abs(imbal_cost_p_6)/imbalance_prosumer
    

#%% pics
# prosumers' costs per unit of imbalance accross scenarios
fig_perunit_cost = plt.figure(1,figsize=[10,6])
ax1, = plt.plot(imbal_cost_p_perunit_imbal_1,label='scenario 1F')
ax2, = plt.plot(imbal_cost_p_perunit_imbal_2,label='scenario 1D')
ax3, = plt.plot(imbal_cost_p_perunit_imbal_3,label='scenario 2F')
ax4, = plt.plot(imbal_cost_p_perunit_imbal_4,label='scenario 2D')
ax5, = plt.plot(imbal_cost_p_perunit_imbal_5,label='scenario 3F',linestyle='dashed')
ax6, = plt.plot(imbal_cost_p_perunit_imbal_6,label='scenario 3D',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4, ax5, ax6], ['scenario 1F', 'scenario 1D','scenario 2F', 'scenario 2D','scenario 3F', 'scenario 3D'])
plt.xlabel('time [15 mins intervals]')
plt.ylabel('cost per unit of imbalance [AUD/kWh]')
plt.title('prosumers\' costs per unit of imbalance accross scenarios')
plt.savefig('figures/perunit_cost_scenarios.png', bbox_inches='tight')
# production costs prosumers
fig_perunit_cost = plt.figure(2,figsize=[10,6])
ax1, = plt.plot(y0_g_bal[:192,7],label='prosumer 7, intercept')
ax2, = plt.plot(y0_g_bal[:192,6],label='prosumer 6, intercept',linestyle='dashed')
ax3, = plt.plot(mm_g_bal[:192,7],label='prosumer 7, ang. coefficient')
ax4, = plt.plot(mm_g_bal[:192,6],label='prosumer 6, ang. coefficient',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4], ['prosumer 7, intercept', 'prosumer 6, intercept','prosumer 7, ang. coefficient', 'prosumer 6, ang. coefficient'])
plt.xlabel('time [15 mins intervals]')
plt.ylabel('production cost [AUD/kWh]')
plt.title('production costs prosumers #6 and #7')
plt.savefig('figures/cost6n7.png', bbox_inches='tight')
# consumption utility prosumers
fig_perunit_cost = plt.figure(3,figsize=[10,6])
ax1, = plt.plot(y0_c_bal[:192,7],label='prosumer 7, intercept')
ax2, = plt.plot(y0_c_bal[:192,6],label='prosumer 6, intercept',linestyle='dashed')
ax3, = plt.plot(mm_c_bal[:192,7],label='prosumer 7, ang. coefficient')
ax4, = plt.plot(mm_c_bal[:192,6],label='prosumer 6, ang. coefficient',linestyle='dashed')
plt.legend([ax1, ax2, ax3, ax4], ['prosumer 7, intercept', 'prosumer 6, intercept','prosumer 7, ang. coefficient', 'prosumer 6, ang. coefficient'])
plt.xlabel('time [15 mins intervals]')
plt.ylabel('consumption utility [AUD/kWh]')
plt.title('consumption utility prosumers #6 and #7')
plt.savefig('figures/utility6n7.png', bbox_inches='tight')
# total costs accross scenarios
imbal_cost_135 = np.array((imbal_cost_1,imbal_cost_3,imbal_cost_5))
imbal_cost_246 = np.array((imbal_cost_2,imbal_cost_4,imbal_cost_6))
xlab = ['Scenario 1','Scenario 2','Scenario 3']
fig = plt.figure(4,figsize=[10,6])
ax = fig.add_subplot(111)
res = pd.DataFrame([imbal_cost_135, imbal_cost_246],index=['Fixed tariff', 'Dynamic tariff'],columns=xlab).transpose()
df_plot = pd.DataFrame([imbal_cost_135, imbal_cost_246],index=['Fixed tariff', 'Dynamic tariff'],columns=xlab).transpose()
df_plot.plot(kind='bar',ax=ax)
plt.xticks(rotation=360)
plt.ylabel('Total imbalance costs [AUD/kWh]')
plt.tight_layout()
plt.grid()
plt.savefig('figures/total_imbal_barchart.png', bbox_inches='tight')

#%% ADMM optimisation BAL via master-sub classes - it should be OK - check the attributes
from ADMM_master_bal import ADMM_Master_bal
o = 0.01
e = 0.00001
asynch = -100
weight = 1/n

# redifining names (so I dont have to change all the names in the function)

Pmin = Pmin_bal
Pmax = Pmax_bal
# Load and Flex_load? and PV? do we want the final result or the variation needed? ask Fabio

#% Cost of Import/Export not considered ==> case = 0
Mast_bal = ADMM_Master_bal(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, deltaPV, deltaLoad, tau, el_price_e, o, e, TMST, window, 0, asynch)
ADMM_bal_summary = Mast_bal.results
ADMM_bal_prices = Mast_bal.prices
ADMM_bal_flag = Mast_bal.flag

#%% - it should be OK
max_l = abs(Mast_bal.variables.l_k - np.transpose(CT_l_sol_bal)).max()
max_p = abs(Mast_bal.variables.p_k - np.transpose(CT_p_sol_bal)).max()
max_q = abs(Mast_bal.variables.q_k - np.transpose(CT_q_sol_bal)).max()
max_alfa = abs(Mast_bal.variables.alfa_k - np.transpose(CT_alfa_sol_bal)).max()
max_beta = abs(Mast_bal.variables.beta_k - np.transpose(CT_beta_sol_bal)).max()
l_1 = Mast_bal.variables.l_k 
p_1 = Mast_bal.variables.p_k
q_1 = Mast_bal.variables.q_k
q_pos_1 = Mast_bal.variables.q_pos_k
alfa_1 = Mast_bal.variables.alfa_k 
beta_1 = Mast_bal.variables.beta_k 
imp_1 = alfa_1.sum(axis=1) 
exp_1 = beta_1.sum(axis=1)
ie_1 = imp_1-exp_1

opt1 = np.empty(TMST)
ooo = np.empty(TMST)
for i1 in range(TMST):
    opt1[i1] = (sum(y0_c[i1,:]*l_1[i1,:] + mm_c[i1,:]/2*l_1[i1,:]*l_1[i1,:]) + sum(y0_g[i1,:]*p_1[i1,:] + mm_g[i1,:]/2*p_1[i1,:]*p_1[i1,:]) #+ el_price_e[i1]*ie_1[i1]
                + 0.001*sum(abs(q_1[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_1[i1,:]) - el_price_e[i1]*sum(beta_1[i1,:]))
    ooo[i1] = (sum(y0_c[i1,:]*CT_l_sol_bal[:,i1] + mm_c[i1,:]/2*CT_l_sol_bal[:,i1]*CT_l_sol_bal[:,i1]) + sum(y0_g_bal[i1,:]*CT_p_sol_bal[:,i1] + mm_g[i1,:]/2*CT_p_sol_bal[:,i1]*CT_p_sol_bal[:,i1]) + 
                (el_price_e[i1]+0.1)*CT_imp_sol_bal[i1] - el_price_e[i1]*CT_exp_sol_bal[i1] + 0.001*sum(abs(CT_q_sol_bal[:,i1]))) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
    
gap_opt1 = opt1-CT_obj_sol
gap_opt12 = opt1-ooo
#%% Save Results
#import shelve
#filename='Res_14_07_1.out'
#var_2_save = ['ADMM_summary','ADMM_prices','ADMM_flag','l_1','p_1','q_1','alfa_1','beta_1']
#if os.path.exists(filename+'.dat'):
#    print('ERROR - file already existing!')
#else:    
#    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
#    for key in var_2_save:
#        try:
#            my_shelf[key] = globals()[key]
#        except TypeError:
#            #
#            # __builtins__, my_shelf, and imported modules can not be shelved.
#            #
#            print(key)
#    my_shelf.close()
#
##%% Community Trading
#CT2_p_sol = np.zeros((g,TMST))
#CT2_l_sol = np.zeros((n,TMST))
#CT2_q_sol = np.zeros((n,TMST))
#CT2_qcomm_sol = np.zeros((3,TMST))
#CT2_alfa_sol = np.zeros((n,TMST))
#CT2_beta_sol = np.zeros((n,TMST))
#CT2_imp_sol = np.zeros(TMST)
#CT2_exp_sol = np.zeros(TMST)
#CT2_obj_sol = np.zeros(TMST)
#CT2_price_sol = np.zeros((n,TMST))
#CT2_price2_sol = np.zeros((2,TMST))
#CT2_price3_sol = np.zeros((3,TMST))
#
#for t in np.arange(0,TMST,24):
#    temp = range(t,t+24)
#    # Create a new model
#    CT2_m = gb.Model("qp")
#    CT2_m.setParam( 'OutputFlag', False )
#    # Create variables
#    p = np.array([CT2_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
#    l = np.array([CT2_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
#    q_imp = np.array([CT2_m.addVar() for k in range(24)])
#    q_exp = np.array([CT2_m.addVar() for k in range(24)])
#    alfa = np.array([CT2_m.addVar() for i in range(n) for k in range(24)])
#    beta = np.array([CT2_m.addVar() for i in range(n) for k in range(24)])
#    q_pos = np.array([CT2_m.addVar() for i in range(n) for k in range(24)])
#    q = np.array([CT2_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
#    q_12 = np.array([CT2_m.addVar() for k in range(24)])
#    q_13 = np.array([CT2_m.addVar() for k in range(24)])
#    q_23 = np.array([CT2_m.addVar() for k in range(24)])
#    q_cpp = np.array([CT2_m.addVar() for i in range(3) for k in range(24)])
#    gamma_i = (el_price_e[temp] + 0.1)
#    gamma_e = -el_price_e[temp]
#    gamma_12 = 0.03
#    gamma_23 = 0.05
#    gamma_13 = 0.06
#    gamma_geo = np.array([gamma_12,gamma_13,gamma_23])
#    CT2_m.update()
#    
#    p = np.transpose(p.reshape(n,24))
#    l = np.transpose(l.reshape(n,24))
#    q = np.transpose(q.reshape(n,24))
#    q_pos = np.transpose(q_pos.reshape(n,24))
#    q_cpp = np.transpose(q_cpp.reshape(3,24))
#    alfa = np.transpose(alfa.reshape(n,24))
#    beta = np.transpose(beta.reshape(n,24))
#    
#    # Set objective: 
#    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
#           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(q_cpp[:,0]*gamma_geo[0]+q_cpp[:,1]*gamma_geo[1]+q_cpp[:,2]*gamma_geo[2]) + sum(0.001*sum(q_pos)))
#    CT2_m.setObjective(obj)
#    # Add constraint
#    for k in range(24):
#        CT2_m.addConstr(sum(q[k,range(0,5)]) - q_12[k] - q_13[k] == 0, name="comm1[%s]"%(k))
#        CT2_m.addConstr(sum(q[k,range(5,10)]) + q_12[k] - q_23[k] == 0, name="comm2[%s]"%(k))
#        CT2_m.addConstr(sum(q[k,range(10,15)]) + q_13[k] + q_23[k] == 0, name="comm3[%s]"%(k))
#        for i in range(n):
#            CT2_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
#            CT2_m.addConstr(q[k,i] <= q_pos[k,i])
#            CT2_m.addConstr(q[k,i] >= -q_pos[k,i])
#        CT2_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
#        CT2_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
#        CT2_m.addConstr(q_12[k] <= q_cpp[k,0])
#        CT2_m.addConstr(q_12[k] >= -q_cpp[k,0])
#        CT2_m.addConstr(q_13[k] <= q_cpp[k,1])
#        CT2_m.addConstr(q_13[k] >= -q_cpp[k,1])
#        CT2_m.addConstr(q_23[k] <= q_cpp[k,2])
#        CT2_m.addConstr(q_23[k] >= -q_cpp[k,2])
#    for i in range(n):
#        CT2_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
##        for j in np.arange(0,24,window):
##            CT5_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
#    CT2_m.update()    
#        
#    CT2_m.optimize()
#    for k in range(24):
#        for i in range(n):
#            CT2_price_sol[i,t+k] = CT2_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
#            CT2_p_sol[i,t+k] = p[k,i].x
#            CT2_l_sol[i,t+k] = l[k,i].x
#            CT2_q_sol[i,t+k] = q[k,i].x
#            CT2_alfa_sol[i,t+k] = alfa[k,i].x
#            CT2_beta_sol[i,t+k] = beta[k,i].x
#        CT2_price2_sol[0,t+k] = CT2_m.getConstrByName("imp_bal[%s]"%k).Pi
#        CT2_price2_sol[1,t+k] = CT2_m.getConstrByName("exp_bal[%s]"%k).Pi
#        CT2_price3_sol[0,t+k] = CT2_m.getConstrByName("comm1[%s]"%(k)).Pi
#        CT2_price3_sol[1,t+k] = CT2_m.getConstrByName("comm2[%s]"%(k)).Pi
#        CT2_price3_sol[2,t+k] = CT2_m.getConstrByName("comm3[%s]"%(k)).Pi
#        CT2_imp_sol[t+k] = q_imp[k].x
#        CT2_exp_sol[t+k] = q_exp[k].x
#        CT2_qcomm_sol[0,t+k] = q_12[k].x
#        CT2_qcomm_sol[1,t+k] = q_13[k].x
#        CT2_qcomm_sol[2,t+k] = q_23[k].x
#
#    del CT2_m
#
#CT2_IE_sol = CT2_imp_sol - CT2_exp_sol
#
##%% ADMM optimisation via master-sub classes
#from ADMM_master_geo3Z import ADMM_Master
#o = 0.01
#e = 0.00001
#asynch = -100
#weight = 1/n
#
#gamma_12 = 0.03
#gamma_13 = 0.06
#gamma_23 = 0.05
#gamma_geo = np.array([gamma_12,gamma_13,gamma_23])
#
#
#Mast2 = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch, gamma_geo)
#ADMM_summary2 = Mast2.results
#ADMM_prices2 = Mast2.prices
#ADMM_flag2 = Mast2.flag
#
##%%
#max_l_2 = abs(Mast2.variables.l_k - np.transpose(CT2_l_sol)).max()
#max_p_2 = abs(Mast2.variables.p_k - np.transpose(CT2_p_sol)).max()
#max_q_2 = abs(Mast2.variables.q_k - np.transpose(CT2_q_sol)).max()
#max_alfa_2 = abs(Mast2.variables.alfa_k - np.transpose(CT2_alfa_sol)).max()
#max_beta_2 = abs(Mast2.variables.beta_k - np.transpose(CT2_beta_sol)).max()
#l_2 = Mast2.variables.l_k 
#p_2 = Mast2.variables.p_k
#q_2 = Mast2.variables.q_k
#alfa_2 = Mast2.variables.alfa_k 
#beta_2 = Mast2.variables.beta_k 
#q2_com = Mast2.variables.pow_comm
#
#imp_2 = alfa_2.sum(axis=1) 
#exp_2 = beta_2.sum(axis=1)
#ie_2 = imp_2-exp_2
#
#opt2 = np.empty(TMST)
#ooo2= np.empty(TMST)
#for i1 in range(TMST):
#    opt2[i1] = (sum(y0_c[i1,:]*l_2[i1,:] + mm_c[i1,:]/2*l_2[i1,:]*l_2[i1,:]) + sum(y0_g[i1,:]*p_2[i1,:] + mm_g[i1,:]/2*p_2[i1,:]*p_2[i1,:]) #+ el_price_e[i1]*ie_1[i1]
#                + 0.001*sum(abs(q_2[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_2[i1,:]) - el_price_e[i1]*sum(beta_2[i1,:]) + sum(gamma_geo*abs(q2_com[i1,:])))
#    ooo2[i1] = (sum(y0_c[i1,:]*CT2_l_sol[:,i1] + mm_c[i1,:]/2*CT2_l_sol[:,i1]*CT2_l_sol[:,i1]) + sum(y0_g[i1,:]*CT2_p_sol[:,i1] + mm_g[i1,:]/2*CT2_p_sol[:,i1]*CT2_p_sol[:,i1]) + 
#                (el_price_e[i1]+0.1)*CT2_imp_sol[i1] - el_price_e[i1]*CT2_exp_sol[i1] + 0.001*sum(abs(CT2_q_sol[:,i1]))  + sum(gamma_geo*abs(CT2_qcomm_sol[:,i1]))) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
#    
#gap_opt2 = opt2-CT2_obj_sol
#gap_opt22 = opt2-ooo2
#
##%% Save Results
#import shelve
#filename='Res_14_07_2.out'
#var_2_save = ['ADMM_summary2','ADMM_prices2','ADMM_flag2','l_2','p_2','q_2','alfa_2','beta_2','q2_com']
#if os.path.exists(filename+'.dat'):
#    print('ERROR - file already existing!')
#else:    
#    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
#    for key in var_2_save:
#        try:
#            my_shelf[key] = globals()[key]
#        except TypeError:
#            #
#            # __builtins__, my_shelf, and imported modules can not be shelved.
#            #
#            print(key)
#    my_shelf.close()
#
##%% Community Trading
#CT3_p_sol = np.zeros((g,TMST))
#CT3_l_sol = np.zeros((n,TMST))
#CT3_q_sol = np.zeros((n,TMST))
#CT3_qcomm_sol = np.zeros(TMST)
#CT3_alfa_sol = np.zeros((n,TMST))
#CT3_beta_sol = np.zeros((n,TMST))
#CT3_imp_sol = np.zeros(TMST)
#CT3_exp_sol = np.zeros(TMST)
#CT3_obj_sol = np.zeros(TMST)
#CT3_price_sol = np.zeros((n,TMST))
#CT3_price2_sol = np.zeros((2,TMST))
#CT3_price3_sol = np.zeros((3,TMST))
#
#for t in np.arange(0,TMST,24):
#    temp = range(t,t+24)
#    # Create a new model
#    CT3_m = gb.Model("qp")
#    CT3_m.setParam( 'OutputFlag', False )
#    # Create variables
#    p = np.array([CT3_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
#    l = np.array([CT3_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
#    q_imp = np.array([CT3_m.addVar() for k in range(24)])
#    q_exp = np.array([CT3_m.addVar() for k in range(24)])
#    alfa = np.array([CT3_m.addVar() for i in range(n) for k in range(24)])
#    beta = np.array([CT3_m.addVar() for i in range(n) for k in range(24)])
#    q_pos = np.array([CT3_m.addVar() for i in range(n) for k in range(24)])
#    q = np.array([CT3_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
#    q_12 = np.array([CT3_m.addVar() for k in range(24)])
#    q_cpp = np.array([CT3_m.addVar() for k in range(24)])
#    gamma_i = (el_price_e[temp] + 0.1)
#    gamma_e = -el_price_e[temp]
#    gamma_12 = 0.03
#    CT3_m.update()
#    
#    p = np.transpose(p.reshape(n,24))
#    l = np.transpose(l.reshape(n,24))
#    q = np.transpose(q.reshape(n,24))
#    q_pos = np.transpose(q_pos.reshape(n,24))
#    alfa = np.transpose(alfa.reshape(n,24))
#    beta = np.transpose(beta.reshape(n,24))
#    
#    # Set objective: 
#    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
#           + sum(gamma_i*q_imp + gamma_e*q_exp) + sum(gamma_12*q_cpp) + sum(0.001*sum(q_pos)))
#    CT3_m.setObjective(obj)
#    # Add constraint
#    for k in range(24):
#        CT3_m.addConstr(sum(q[k,range(0,10)]) - q_12[k] == 0, name="comm1[%s]"%(k))
#        CT3_m.addConstr(sum(q[k,range(10,15)]) + q_12[k] == 0, name="comm2[%s]"%(k))
#        for i in range(n):
#            CT3_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
#            CT3_m.addConstr(q[k,i] <= q_pos[k,i])
#            CT3_m.addConstr(q[k,i] >= -q_pos[k,i])
#        CT3_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
#        CT3_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
#        CT3_m.addConstr(q_12[k] <= q_cpp[k])
#        CT3_m.addConstr(q_12[k] >= -q_cpp[k])
#    for i in range(n):
#        CT3_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
##        for j in np.arange(0,24,window):
##            CT5_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
#    CT3_m.update()    
#        
#    CT3_m.optimize()
#    for k in range(24):
#        for i in range(n):
#            CT3_price_sol[i,t+k] = CT3_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
#            CT3_p_sol[i,t+k] = p[k,i].x
#            CT3_l_sol[i,t+k] = l[k,i].x
#            CT3_q_sol[i,t+k] = q[k,i].x
#            CT3_alfa_sol[i,t+k] = alfa[k,i].x
#            CT3_beta_sol[i,t+k] = beta[k,i].x
#        CT3_price2_sol[0,t+k] = CT3_m.getConstrByName("imp_bal[%s]"%k).Pi
#        CT3_price2_sol[1,t+k] = CT3_m.getConstrByName("exp_bal[%s]"%k).Pi
#        CT3_price3_sol[0,t+k] = CT3_m.getConstrByName("comm1[%s]"%(k)).Pi
#        CT3_price3_sol[1,t+k] = CT3_m.getConstrByName("comm2[%s]"%(k)).Pi
#        CT3_imp_sol[t+k] = q_imp[k].x
#        CT3_exp_sol[t+k] = q_exp[k].x
#        CT3_qcomm_sol[t+k] = q_12[k].x
#
#    del CT3_m
#    
#CT3_IE_sol = CT3_imp_sol - CT3_exp_sol
#sum_q = CT3_q_sol.sum(axis=0)
#
##%% ADMM optimisation via master-sub classes
#from ADMM_master_geo2Z import ADMM_Master
#o = 0.01
#e = 0.00001
#asynch = -100
#weight = 1/n
#
#gamma_12 = 0.03
#gamma_13 = 0.06
#gamma_23 = 0.05
#gamma_geo = gamma_12 #np.array([gamma_12,gamma_13,gamma_23])
#
#
#Mast3 = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch, gamma_geo)
#ADMM_summary3 = Mast3.results
#ADMM_prices3 = Mast3.prices
#ADMM_flag3 = Mast3.flag
#
##%%
#max_l_3 = abs(Mast3.variables.l_k - np.transpose(CT3_l_sol)).max()
#max_p_3 = abs(Mast3.variables.p_k - np.transpose(CT3_p_sol)).max()
#max_q_3 = abs(Mast3.variables.q_k - np.transpose(CT3_q_sol)).max()
#max_alfa_3 = abs(Mast3.variables.alfa_k - np.transpose(CT3_alfa_sol)).max()
#max_beta_3 = abs(Mast3.variables.beta_k - np.transpose(CT3_beta_sol)).max()
#l_3 = Mast3.variables.l_k 
#p_3 = Mast3.variables.p_k
#q_3 = Mast3.variables.q_k
#alfa_3 = Mast3.variables.alfa_k 
#beta_3 = Mast3.variables.beta_k 
#q3_com = Mast3.variables.pow_comm
#
#imp_3 = alfa_3.sum(axis=1) 
#exp_3 = beta_3.sum(axis=1)
#ie_3 = imp_3-exp_3
#
#opt3 = np.empty(TMST)
#ooo3= np.empty(TMST)
#for i1 in range(TMST):
#    opt3[i1] = (sum(y0_c[i1,:]*l_3[i1,:] + mm_c[i1,:]/2*l_3[i1,:]*l_3[i1,:]) + sum(y0_g[i1,:]*p_3[i1,:] + mm_g[i1,:]/2*p_3[i1,:]*p_3[i1,:]) #+ el_price_e[i1]*ie_1[i1]
#                + 0.001*sum(abs(q_3[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_3[i1,:]) - el_price_e[i1]*sum(beta_3[i1,:]) + gamma_geo*abs(q3_com[i1]))
#    ooo3[i1] = (sum(y0_c[i1,:]*CT3_l_sol[:,i1] + mm_c[i1,:]/2*CT3_l_sol[:,i1]*CT3_l_sol[:,i1]) + sum(y0_g[i1,:]*CT3_p_sol[:,i1] + mm_g[i1,:]/2*CT3_p_sol[:,i1]*CT3_p_sol[:,i1]) + 
#                (el_price_e[i1]+0.1)*CT3_imp_sol[i1] - el_price_e[i1]*CT3_exp_sol[i1] + 0.001*sum(abs(CT3_q_sol[:,i1]))  + gamma_geo*abs(CT3_qcomm_sol[i1])) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
#    
#gap_opt3 = opt3-CT3_obj_sol
#gap_opt32 = opt3-ooo3
#
#
##%% Save Results
#import shelve
#filename='Res_14_07_3.out'
#var_2_save = ['ADMM_summary3','ADMM_prices3','ADMM_flag3','l_3','p_3','q_3','alfa_3','beta_3','q3_com']
#if os.path.exists(filename+'.dat'):
#    print('ERROR - file already existing!')
#else:    
#    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
#    for key in var_2_save:
#        try:
#            my_shelf[key] = globals()[key]
#        except TypeError:
#            #
#            # __builtins__, my_shelf, and imported modules can not be shelved.
#            #
#            print(key)
#    my_shelf.close()
#
##%% Community Trading
#CT4_p_sol = np.zeros((g,TMST))
#CT4_l_sol = np.zeros((n,TMST))
#CT4_q_sol = np.zeros((n,TMST))
#CT4_alfa_sol = np.zeros((n,TMST))
#CT4_beta_sol = np.zeros((n,TMST))
#CT4_imp_sol = np.zeros(TMST)
#CT4_exp_sol = np.zeros(TMST)
#CT4_obj_sol = np.zeros(TMST)
#CT4_price_sol = np.zeros((n,TMST))
#CT4_price2_sol = np.zeros((3,TMST))
#CT4_price3_sol = np.zeros((n,TMST))
#CT4_theta_sol = np.zeros(TMST)
#
#for t in np.arange(0,TMST,24):
#    temp = range(t,t+24)
#    # Create a new model
#    CT4_m = gb.Model("qp")
#    CT4_m.setParam( 'OutputFlag', False )
#    # Create variables
#    p = np.array([CT4_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
#    l = np.array([CT4_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
#    q_imp = np.array([CT4_m.addVar() for k in range(24)])
#    q_exp = np.array([CT4_m.addVar() for k in range(24)])
#    alfa = np.array([CT4_m.addVar() for i in range(n) for k in range(24)])
#    beta = np.array([CT4_m.addVar() for i in range(n) for k in range(24)])
#    q_pos = np.array([CT4_m.addVar() for i in range(n) for k in range(24)])
#    q = np.array([CT4_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
#    theta = np.array([CT4_m.addVar() for k in range(24)])
#    gamma_i = (el_price_e[temp] + 0.1)
#    gamma_e = -el_price_e[temp]
#    gamma_m = 1.0
#    CT4_m.update()
#    
#    p = np.transpose(p.reshape(n,24))
#    l = np.transpose(l.reshape(n,24))
#    q = np.transpose(q.reshape(n,24))
#    q_pos = np.transpose(q_pos.reshape(n,24))
#    alfa = np.transpose(alfa.reshape(n,24))
#    beta = np.transpose(beta.reshape(n,24))
#    
#    # Set objective: 
#    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
#           + sum(gamma_i*q_imp + gamma_e*q_exp + gamma_m*theta) + sum(0.001*sum(q_pos)))
#    CT4_m.setObjective(obj)
#    # Add constraint
#    for k in range(24):
#        CT4_m.addConstr(sum(q[k,:]) == 0, name="comm[%s]"%(k))
#        for i in range(n):
#            CT4_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
#            CT4_m.addConstr(q[k,i] <= q_pos[k,i])
#            CT4_m.addConstr(q[k,i] >= -q_pos[k,i])
#            CT4_m.addConstr(alfa[k,i] - theta[k] <= 0, name="maxx[%s,%s]"% (k,i))
#        CT4_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
#        CT4_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
#    for i in range(n):
#        CT4_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
##        for j in np.arange(0,24,window):
##            CT5_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
#    CT4_m.update()    
#        
#    CT4_m.optimize()
#    for k in range(24):
#        for i in range(n):
#            CT4_price_sol[i,t+k] = CT4_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
#            CT4_price3_sol[i,t+k] = CT4_m.getConstrByName("maxx[%s,%s]"%(k,i)).Pi
#            CT4_p_sol[i,t+k] = p[k,i].x
#            CT4_l_sol[i,t+k] = l[k,i].x
#            CT4_q_sol[i,t+k] = q[k,i].x
#            CT4_alfa_sol[i,t+k] = alfa[k,i].x
#            CT4_beta_sol[i,t+k] = beta[k,i].x
#        CT4_price2_sol[0,t+k] = CT4_m.getConstrByName("comm[%s]"%k).Pi
#        CT4_price2_sol[1,t+k] = CT4_m.getConstrByName("imp_bal[%s]"%k).Pi
#        CT4_price2_sol[2,t+k] = CT4_m.getConstrByName("exp_bal[%s]"%k).Pi
#        CT4_imp_sol[t+k] = q_imp[k].x
#        CT4_exp_sol[t+k] = q_exp[k].x
#        CT4_theta_sol[t+k] = theta[k].x
#
#    del CT4_m
#
#CT4_IE_sol = CT4_imp_sol - CT4_exp_sol
#
##%% ADMM optimisation via master-sub classes
#from ADMM_master_max import ADMM_Master
#o = 0.01
#e = 0.00001
#asynch = -100
#weight = 1/n
#
#Mast4 = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch)
#ADMM_summary4 = Mast4.results
#ADMM_prices4 = Mast4.prices
#ADMM_flag4 = Mast4.flag
#
##%%
#max_l_4 = abs(Mast4.variables.l_k - np.transpose(CT4_l_sol)).max()
#max_p_4 = abs(Mast4.variables.p_k - np.transpose(CT4_p_sol)).max()
#max_q_4 = abs(Mast4.variables.q_k - np.transpose(CT4_q_sol)).max()
#max_alfa_4 = abs(Mast4.variables.alfa_k - np.transpose(CT4_alfa_sol)).max()
#max_beta_4 = abs(Mast4.variables.beta_k - np.transpose(CT4_beta_sol)).max()
#l_4 = Mast4.variables.l_k 
#p_4 = Mast4.variables.p_k
#q_4 = Mast4.variables.q_k
#alfa_4 = Mast4.variables.alfa_k 
#beta_4 = Mast4.variables.beta_k 
#theta = Mast4.variables.pow_theta
#
#imp_4 = alfa_4.sum(axis=1) 
#exp_4 = beta_4.sum(axis=1)
#ie_4 = imp_4-exp_4
#
#opt4 = np.empty(TMST)
#ooo4= np.empty(TMST)
#for i1 in range(TMST):
#    opt4[i1] = (sum(y0_c[i1,:]*l_4[i1,:] + mm_c[i1,:]/2*l_4[i1,:]*l_4[i1,:]) + sum(y0_g[i1,:]*p_4[i1,:] + mm_g[i1,:]/2*p_4[i1,:]*p_4[i1,:]) #+ el_price_e[i1]*ie_1[i1]
#                + 0.001*sum(abs(q_4[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_4[i1,:]) - el_price_e[i1]*sum(beta_4[i1,:]) + 1.0*theta[i1])
#    ooo4[i1] = (sum(y0_c[i1,:]*CT4_l_sol[:,i1] + mm_c[i1,:]/2*CT4_l_sol[:,i1]*CT4_l_sol[:,i1]) + sum(y0_g[i1,:]*CT4_p_sol[:,i1] + mm_g[i1,:]/2*CT4_p_sol[:,i1]*CT4_p_sol[:,i1]) + 
#                (el_price_e[i1]+0.1)*CT4_imp_sol[i1] - el_price_e[i1]*CT4_exp_sol[i1] + 0.001*sum(abs(CT4_q_sol[:,i1]))  + 1.0*CT4_theta_sol[i1]) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
#    
#gap_opt4 = opt4-CT4_obj_sol
#gap_opt42 = opt4-ooo4
#
##%% Save Results
#import shelve
#filename='Res_14_07_4.out'
#var_2_save = ['ADMM_summary4','ADMM_prices4','ADMM_flag4','l_4','p_4','q_4','alfa_4','beta_4','theta']
#if os.path.exists(filename+'.dat'):
#    print('ERROR - file already existing!')
#else:    
#    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
#    for key in var_2_save:
#        try:
#            my_shelf[key] = globals()[key]
#        except TypeError:
#            #
#            # __builtins__, my_shelf, and imported modules can not be shelved.
#            #
#            print(key)
#    my_shelf.close()
#    
##%% Community Trading
#CT5_p_sol = np.zeros((g,TMST))
#CT5_l_sol = np.zeros((n,TMST))
#CT5_q_sol = np.zeros((n,TMST))
#CT5_alfa_sol = np.zeros((n,TMST))
#CT5_beta_sol = np.zeros((n,TMST))
#CT5_imp_sol = np.zeros(TMST)
#CT5_exp_sol = np.zeros(TMST)
#CT5_price_sol = np.zeros((n,TMST))
#CT5_price2_sol = np.zeros((3,TMST))
#CT5_price3_sol = np.zeros(TMST)
#CT5_theta_sol = np.zeros(TMST)
#
#for t in np.arange(0,TMST,24):
#    temp = range(t,t+24)
#    # Create a new model
#    CT5_m = gb.Model("qp")
#    CT5_m.setParam( 'OutputFlag', False )
#    # Create variables
#    p = np.array([CT5_m.addVar(lb=Pmin[i]+PV[t+k,i], ub=Pmax[i]+PV[t+k,i]) for i in range(n) for k in range(24)])
#    l = np.array([CT5_m.addVar(lb=Lmin[t+k,i], ub=Lmax[t+k,i]) for i in range(n) for k in range(24)])
#    q_imp = np.array([CT5_m.addVar() for k in range(24)])
#    q_exp = np.array([CT5_m.addVar() for k in range(24)])
#    alfa = np.array([CT5_m.addVar() for i in range(n) for k in range(24)])
#    beta = np.array([CT5_m.addVar() for i in range(n) for k in range(24)])
##    delta = np.array([CT5_m.addVar() for k in range(24)])
#    q_pos = np.array([CT5_m.addVar() for i in range(n) for k in range(24)])
#    q = np.array([CT5_m.addVar(lb = -gb.GRB.INFINITY) for i in range(n) for k in range(24)])
#    theta = CT5_m.addVar()
#    gamma_i = (el_price_e[temp] + 0.1)
#    gamma_e = -el_price_e[temp]
#    gamma_m = 1.0
#    CT5_m.update()
#    
#    p = np.transpose(p.reshape(n,24))
#    l = np.transpose(l.reshape(n,24))
#    q = np.transpose(q.reshape(n,24))
#    q_pos = np.transpose(q_pos.reshape(n,24))
#    alfa = np.transpose(alfa.reshape(n,24))
#    beta = np.transpose(beta.reshape(n,24))
#    
#    # Set objective: 
#    obj = (sum(sum(y0_c[temp,:]*l + mm_c[temp,:]/2*l*l) + sum(y0_g[temp,:]*p + mm_g[temp,:]/2*p*p)) 
#           + sum(gamma_i*q_imp + gamma_e*q_exp) + gamma_m*theta + sum(0.001*sum(q_pos)))
#    CT5_m.setObjective(obj)
#    # Add constraint
#    for k in range(24):
#        CT5_m.addConstr(1*sum(q[k,:]) == 0, name="comm[%s]"%(k))
#        for i in range(n):
#            CT5_m.addConstr(p[k,i] + l[k,i] + q[k,i] + alfa[k,i] - beta[k,i] == 0, name="pros[%s,%s]"% (k,i))
#            CT5_m.addConstr(q[k,i] <= q_pos[k,i])
#            CT5_m.addConstr(q[k,i] >= -q_pos[k,i])
#        CT5_m.addConstr(sum(alfa[k,:]) - q_imp[k] == 0, name="imp_bal[%s]"%(k))
#        CT5_m.addConstr(sum(beta[k,:]) - q_exp[k] == 0, name="exp_bal[%s]"%(k))
##        CT5_m.addConstr(1*(sum(alfa[k,:]) - delta[k]) == 0, name="max_bal[%s]"%(k))
##        CT5_m.addConstr(delta[k] - theta <= 0, name="maxx[%s]"%(k))
#        CT5_m.addConstr(q_imp[k] - theta <= 0, name="maxx[%s]"%(k)) 
#    for i in range(n):
#        CT5_m.addConstr(sum(Agg_load[temp,i] + l[:,i]) == 0)
##        for j in np.arange(0,24,window):
##            CT5_m.addConstr(sum(Agg_load[range(t+j,t+j+window),i] + l[range(j,j+window),i]) == 0)
#    CT5_m.update()    
#        
#    CT5_m.optimize()
#    for k in range(24):
#        for i in range(n):
#            CT5_price_sol[i,t+k] = CT5_m.getConstrByName("pros[%s,%s]"%(k,i)).Pi
#            CT5_p_sol[i,t+k] = p[k,i].x
#            CT5_l_sol[i,t+k] = l[k,i].x
#            CT5_q_sol[i,t+k] = q[k,i].x
#            CT5_alfa_sol[i,t+k] = alfa[k,i].x
#            CT5_beta_sol[i,t+k] = beta[k,i].x
#        CT5_price2_sol[0,t+k] = CT5_m.getConstrByName("comm[%s]"%k).Pi
#        CT5_price2_sol[1,t+k] = CT5_m.getConstrByName("imp_bal[%s]"%k).Pi
#        CT5_price2_sol[2,t+k] = CT5_m.getConstrByName("exp_bal[%s]"%k).Pi
#        CT5_price3_sol[t+k] = CT5_m.getConstrByName("maxx[%s]"%k).Pi
#        CT5_imp_sol[t+k] = q_imp[k].x
#        CT5_exp_sol[t+k] = q_exp[k].x
#        CT5_theta_sol[t+k] = theta.x
#
#    del CT5_m
#
#CT5_IE_sol = CT5_imp_sol - CT5_exp_sol
#sum_q = CT5_q_sol.sum(axis=0)
#
##%% ADMM optimisation via master-sub classes
#from ADMM_master_peak2 import ADMM_Master
#o = 0.01
#e = 0.00001
#asynch = -100
#weight = 1/n
#
#Mast5 = ADMM_Master(b1, c1, Pmin, Pmax, PV, b2, c2, Load, Flex_load, tau, el_price_e, o, e, TMST, window, 0, asynch)
#ADMM_summary5 = Mast5.results
#ADMM_prices5 = Mast5.prices
#
##%%
#max_l_5 = abs(Mast5.variables.l_k - np.transpose(CT5_l_sol)).max()
#max_p_5 = abs(Mast5.variables.p_k - np.transpose(CT5_p_sol)).max()
#max_q_5 = abs(Mast5.variables.q_k - np.transpose(CT5_q_sol)).max()
#max_alfa_5 = abs(Mast5.variables.alfa_k - np.transpose(CT5_alfa_sol)).max()
#max_beta_5 = abs(Mast5.variables.beta_k - np.transpose(CT5_beta_sol)).max()
#l_5 = Mast5.variables.l_k 
#p_5 = Mast5.variables.p_k
#q_5 = Mast5.variables.q_k
#alfa_5 = Mast5.variables.alfa_k 
#beta_5 = Mast5.variables.beta_k 
#theta_5 = Mast5.variables.pow_theta
#
#imp_5 = alfa_5.sum(axis=1) 
#exp_5 = beta_5.sum(axis=1)
#ie_5 = imp_5-exp_5
#
#opt5 = np.empty(TMST)
#ooo5= np.empty(TMST)
#for i1 in range(TMST):
#    opt5[i1] = (sum(y0_c[i1,:]*l_5[i1,:] + mm_c[i1,:]/2*l_5[i1,:]*l_5[i1,:]) + sum(y0_g[i1,:]*p_5[i1,:] + mm_g[i1,:]/2*p_5[i1,:]*p_5[i1,:]) #+ el_price_e[i1]*ie_1[i1]
#                + 0.001*sum(abs(q_5[i1,:])) + (el_price_e[i1]+0.1)*sum(alfa_5[i1,:]) - el_price_e[i1]*sum(beta_5[i1,:]) + 1.0*theta_5[i1])
#    ooo5[i1] = (sum(y0_c[i1,:]*CT5_l_sol[:,i1] + mm_c[i1,:]/2*CT5_l_sol[:,i1]*CT5_l_sol[:,i1]) + sum(y0_g[i1,:]*CT5_p_sol[:,i1] + mm_g[i1,:]/2*CT5_p_sol[:,i1]*CT5_p_sol[:,i1]) + 
#                (el_price_e[i1]+0.1)*CT5_imp_sol[i1] - el_price_e[i1]*CT5_exp_sol[i1] + 0.001*sum(abs(CT5_q_sol[:,i1]))  + 1.0*CT5_theta_sol[i1]) #+ pr_i*sum(CT_alfa_sol[:,i1]) + pr_e*sum(CT_beta_sol[:,i1]))
#    
##gap_opt5 = opt5-CT5_obj_sol
#gap_opt52 = opt5-ooo5
##%% Save Results
#import shelve
#filename='Res_14_07_0.out'
#var_2_save = ['Mast','Mast2','Mast3','Mast4']
#if os.path.exists(filename+'.dat'):
#    print('ERROR - file already existing!')
#else:    
#    my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
#    for key in var_2_save:
#        try:
#            my_shelf[key] = globals()[key]
#        except TypeError:
#            #
#            # __builtins__, my_shelf, and imported modules can not be shelved.
#            #
#            print(key)
#    my_shelf.close()