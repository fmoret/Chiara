# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:16:33 2017

@author: Chiara
"""

# Import Gurobi Library
import gurobipy as gb
import numpy as np
#from math import log10

# Class which can have attributes set.
class expando(object):
    pass

# Optimization class
class ADMM_Pros_bal:
    def __init__(self, MP, idx):
        self.variables = expando()
        self.params = expando()
        
        self.MP = MP
        self.params.idx = idx
        self.Pmin = self.MP.data.Pmin[idx]   
        self.Pmax = self.MP.data.Pmax[idx] 
        self.Lmin = self.MP.data.Lmin[idx]   
        self.Lmax = self.MP.data.Lmax[idx] 
        self.p_tilde = self.MP.data.p_tilde[idx]   
        self.l_tilde = self.MP.data.l_tilde[idx] 
#        self.PV = self.MP.data.PV[:,idx] # forse questo non serve proprio
        self.deltaPV = self.MP.data.deltaPV
        self.deltaLoad = self.MP.data.deltaLoad
#        self.goal = self.MP.data.goal[:,idx] 
        y0_c_DA = self.MP.data.y0_c_DA[:,idx]
        y0_g_DA = self.MP.data.y0_g_DA[:,idx]
        mm_c_DA = self.MP.data.mm_c_DA[:,idx]
        mm_g_DA = self.MP.data.mm_g_DA[:,idx]
        
        
        self.y0_c_bal = y0_c_DA [len(self.l_tilde)]+ mm_c_DA[len(self.l_tilde)]*self.l_tilde
        self.mm_c_bal = mm_c_DA
        self.y0_g_bal = y0_g_DA[len(self.p_tilde)] + mm_g_DA[len(self.p_tilde)]*self.p_tilde
        self.mm_g_bal = mm_g_DA

        self._build_model_()        

    def optimize(self):
        self._build_objective_()
        self.model.optimize()
        x = np.random.normal(0,1)
        if x>self.MP.threshold:
            for k in range(96):
                self.beta_old[k] = self.variables.beta[k].x
                self.alfa_old[k] =self.variables.alfa[k].x  
                self.q_old[k] =self.variables.q[k].x  
                self.pow_old[k] = self.variables.power[k].x
                self.load_old[k] = self.variables.load[k].x
                self.price_old[k] = self.model.getConstrByName("pow_bal[%s]"%k).Pi
        res = np.empty((96,6))
        res[:,0] = self.pow_old
        res[:,1] = self.load_old
        res[:,2] = self.q_old
        res[:,3] = self.alfa_old
        res[:,4] = self.beta_old
        res[:,5] = self.price_old 
        return res
        
    ###
    #   Model Building
    ###
    def _build_model_(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.update()
        
    def _update_model_(self, t):
        self.t = t
        self.beta_old = np.zeros(96)
        self.alfa_old = np.zeros(96)
        self.q_old = np.zeros(96)
        self.pow_old = np.zeros(96)
        self.load_old = np.zeros(96)
        self.price_old = np.zeros(96)
        self._build_variables_()
        self._build_constraint_()
        self._build_objective_()

    def _build_variables_(self,):
        m = self.model
        self.variables.load = np.array([m.addVar(lb=self.Lmin[self.t+k], ub=self.Lmax[self.t+k]) for k in range(96)])
        #check if u need the PV in the next one
        self.variables.power = np.array([m.addVar(lb=self.Pmin[self.t+k], ub=self.Pmax[self.t+k]) for k in range(96)])
        self.variables.beta = np.array([m.addVar() for k in range(96)])
        self.variables.alfa = np.array([m.addVar() for k in range(96)])
        self.variables.q = np.array([m.addVar(-gb.GRB.INFINITY) for k in range(96)])
        self.variables.q_pos = np.array([m.addVar() for k in range(96)])
        m.update()
        
    def _build_constraint_(self,):
        m = self.model
        for k in range(96):
            m.addConstr(self.variables.load[k] + self.variables.power[k] + self.variables.q[k] + self.variables.alfa[k] - self.variables.beta[k] + self.data.deltaPV[k] - self.data.deltaLoad[k] == 0, name="pow_bal[%s]"%k) 
            m.addConstr(self.variables.q[k] <= self.variables.q_pos[k])
            m.addConstr(self.variables.q[k] >= -self.variables.q_pos[k])
#        t = self.MP.temp[0]
#        for j in np.arange(0,24,self.window):
#            m.addConstr(sum(self.goal[range(t+j,t+j+self.window)] - self.variables.load[range(t+j,t+j+self.window)]) == 0)
        #m.addConstr(sum(self.goal[self.MP.temp] - self.variables.load) == 0)
        m.update()

    def _build_objective_(self):
        m = self.model
        temp = self.MP.temp
        y0_c_bal = self.y0_c_bal[temp]
        mm_c_bal = self.mm_c_bal[temp]
        y0_g_bal = self.y0_g_bal[temp]
        mm_g_bal = self.mm_g_bal[temp]
        price0 = self.MP.variables.price_comm[temp]
        price1 = self.MP.variables.price_IE[temp,:]
        o = self.MP.data.sigma
        res0 = self.MP.variables.r_k[range(0,96)] + self.MP.weight*(self.variables.q - self.q_old)
        res1 = self.MP.variables.r_k[range(96,192)] + (self.variables.alfa - self.alfa_old)
        res2 = self.MP.variables.r_k[range(192,288)] + self.variables.beta - self.beta_old
        self.objective = (sum(y0_c_bal*(self.variables.load) + mm_c_bal/2*(self.variables.load*self.variables.load) + y0_g_bal*(self.variables.power) + mm_g_bal/2*(self.variables.power*self.variables.power))
                          + sum(0.001*self.variables.q_pos) + sum(price0*res0) + sum(price1[:,0]*res1+price1[:,1]*res2) + o/2*(sum(res0*res0)+sum(res1*res1+res2*res2))) #+ sum(0.1*self.variables.alfa + 0.05*self.variables.beta 
        m.setObjective(self.objective)
        m.update()
        
    def clean(self):
        del self.model