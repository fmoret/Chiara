# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:08:15 2017

@author: Chiara
"""

import gurobipy as gb
import numpy as np
import sys
from math import sqrt
import pandas as pd

from ADMM_pros_bal import ADMM_Pros_bal

# Class which can have attributes set
class expando(object):
    pass


class ADMM_Master_bal:
    def __init__(self, Pmin, Pmax, Lmin, Lmax, y0_c_DA, mm_c_DA, y0_g_DA, mm_g_DA, deltaPV, deltaLoad, p_tilde, l_tilde, tau, el_price_exp, el_price_imp, o, e, time, window, case, threshold):

        self.data = expando()
        self.variables = expando()

        self.results = expando()
        self.params = expando()
        
        self.data.y0_g_DA = y0_g_DA
        self.data.mm_g_DA = mm_g_DA
        self.data.Pmin = Pmin #already shifted
        self.data.Pmax = Pmax #already shifted
        self.data.Lmin = Lmin #already shifted
        self.data.Lmax = Lmax #already shifted
        self.data.p_tilde = p_tilde
        self.data.l_tilde = l_tilde
        #self.data.PV = PV #this is a matrix of zeros
        self.data.deltaPV = deltaPV
        self.data.deltaLoad = deltaLoad
        self.data.y0_c_DA = y0_c_DA
        self.data.mm_c_DA = mm_c_DA
        #self.data.goal = - (Load + Flex_load)
        self.data.tau = tau
        self.data.el_price_exp = el_price_exp
        self.data.el_price_imp = el_price_imp
        self.data.rho = o
        self.data.tol = e
        
        self.params.num_pros = deltaPV.shape[1]
        self.case = case
        self.window = window
        self.params.time = time
        
        self.variables.p_k = np.zeros((time, self.params.num_pros))
        self.variables.l_k = np.zeros((time, self.params.num_pros))
        self.variables.q_k = np.zeros((time, self.params.num_pros))
        self.variables.q_pos_k = np.zeros((time, self.params.num_pros))
        self.variables.alfa_k = np.zeros((time, self.params.num_pros))
        self.variables.beta_k = np.zeros((time, self.params.num_pros))
        self.variables.pow_imp = np.zeros(time)
        self.variables.pow_exp = np.zeros(time)
        
        self.variables.price_comm = np.zeros(time)
        self.variables.price_IE = np.ones((time,2))
        
        self.imbalance = np.zeros((time+1, self.params.num_pros))
        
        self.weight = 1/self.params.num_pros
        self.threshold = threshold
        
        self._init_model()
        self._init_subproblems()
        self.optimize()
        #self.clean()

    ###
    # Model Building
    ###
    def _init_model(self):
        self.model = gb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.update()
        self.stdold = sys.stdout
#        sys.stdout = open('pros_out.txt','w')

    def _build_variables(self):
        m = self.model
        self.variables.q_imp = np.array([m.addVar() for k in range(96)])
        self.variables.q_exp = np.array([m.addVar() for k in range(96)])
        m.update()

    def _build_objective(self, t):
        m = self.model
        q = self.variables.q_k[self.temp,:]
        alfa = self.variables.alfa_k[self.temp,:]
        beta = self.variables.beta_k[self.temp,:]
        pr_exp = -self.data.el_price_exp[self.temp]
        pr_imp = self.data.el_price_imp[self.temp] # + self.data.tau
        price_0 = self.variables.price_comm[self.temp]
        price_1 = self.variables.price_IE[self.temp,:]
        o = self.data.sigma
        res0 = self.weight*q.sum(axis=1)
        res1 = alfa.sum(axis=1) - self.variables.q_imp
        res2 = beta.sum(axis=1) - self.variables.q_exp
        
        self.objective = sum(pr_imp*self.variables.q_imp + pr_exp*self.variables.q_exp + price_0*res0) + sum(price_1[:,0]*res1+price_1[:,1]*res2) + o/2*(sum(res0*res0)+sum(res1*res1+res2*res2))
        m.setObjective(self.objective)
        m.update()

    def _init_subproblems(self):
        self._init_prosumers()

    def _init_prosumers(self):
        # Only build submodels if they don't exist or a rebuild is forced.
        if not hasattr(self, 'submodels_pros'):
            self.submodels_pros = {i: ADMM_Pros_bal(self, idx=i) for i in range(self.params.num_pros)} #dizionario
                                   
    def _model_update(self, t):
        self._build_variables()
        self._build_objective(t)
        self._pros_update(t)
                                  
    def _pros_update(self, t):
        for i in range(self.params.num_pros):
            self.submodels_pros.get(i)._update_model_(t)
            
    def optimize(self):
        resul = []
        resul2 = []
        fl = []

        for t in np.arange(0,self.params.time,96): #8,self.params.time
            self.temp = range(t,t+96) 
            self.variables.iter = 0
            self.variables.r_k = np.ones(3*96)#+self.params.num_pros)
            real_rk = np.ones(3*96)
            self.variables.s_k = 1.0
            self.data.sigma = self.data.rho
            res = []
            res2 = []
            res3 = []
            res4 = []
            res5 = []
            res6 = []
            res7 = []
            res8 = []
            res9 = []
            res10 = []
            self._model_update(t)
            flag = 0
            stack = 1000*np.ones(100)
            
            while (np.linalg.norm(real_rk)>self.data.tol or self.variables.s_k>self.data.tol) and self.variables.iter<10000 and np.linalg.norm(real_rk)<10**10: #np.linalg.norm
                self.variables.iter = self.variables.iter +1
                
                #Solve subproblems
                for j in range(self.params.num_pros):
                    sol = self.submodels_pros[j].optimize()
                    self.variables.p_k[self.temp,j] = np.copy(sol[:,0]) 
                    self.variables.l_k[self.temp,j] = np.copy(sol[:,1]) 
                    self.variables.q_k[self.temp,j] = np.copy(sol[:,2]) 
                    self.variables.alfa_k[self.temp,j] = np.copy(sol[:,3]) 
                    self.variables.beta_k[self.temp,j] = np.copy(sol[:,4])
                    pp = np.copy(sol[:,5])

                #Solve main problem
                self._build_objective(t)
                self.model.optimize()
                imp_old = self.variables.pow_imp[self.temp]
                exp_old = self.variables.pow_exp[self.temp]
                for k in range(96):
                    self.variables.pow_imp[t+k] = self.variables.q_imp[k].x
                    self.variables.pow_exp[t+k] = self.variables.q_exp[k].x

                #Calculate residulas
                self.variables.r_k[range(0,96)] = self.weight*self.variables.q_k[self.temp,:].sum(axis=1)
                self.variables.r_k[range(96,192)] = (self.variables.alfa_k[self.temp,:].sum(axis=1) - self.variables.pow_imp[self.temp])
                self.variables.r_k[range(192,288)] = self.variables.beta_k[self.temp,:].sum(axis=1) - self.variables.pow_exp[self.temp]

                if max(abs(np.linalg.norm(self.variables.r_k)*np.ones(100)-stack))<0.01*np.linalg.norm(self.variables.r_k) and self.data.sigma<1:
                    self.data.sigma = self.data.sigma*2
                    stack = 100*np.ones(100)
                elif np.linalg.norm(self.variables.r_k)-np.mean(stack)>0.1*np.linalg.norm(self.variables.r_k):
                    self.data.sigma = self.data.sigma/3
                    stack = 100*np.ones(100)
                stack[range(1,100)] = np.copy(stack[range(0,99)])
                stack[0] = np.linalg.norm(self.variables.r_k)

                Dimp = self.variables.pow_imp[self.temp]-imp_old
                Dexp = self.variables.pow_exp[self.temp]-exp_old
                self.variables.s_k = sqrt(self.params.num_pros)*self.data.sigma*sqrt(sum(Dimp*Dimp + Dexp*Dexp))

                real_rk = np.copy(self.variables.r_k)
                real_rk[range(0,96)] = real_rk[range(0,96)]/self.weight
                
                #Update price
                self.variables.price_comm[self.temp] = self.variables.price_comm[self.temp] + self.data.sigma*self.variables.r_k[range(0,96)]
                self.variables.price_IE[self.temp,0] = self.variables.price_IE[self.temp,0] + self.data.sigma*self.variables.r_k[range(96,192)]
                self.variables.price_IE[self.temp,1] = self.variables.price_IE[self.temp,1] + self.data.sigma*self.variables.r_k[range(192,288)]

                print(t, '-',self.variables.iter,'     Primal Residual ', np.linalg.norm(real_rk), '     Dual Residual ', self.variables.s_k, '     Sigma ', self.data.sigma)

                if np.linalg.norm(real_rk)>10**10:
                    flag = 2
                if self.variables.iter==5000:
                    flag = 1
                
                for k in range(96):
                    d = {'imp': self.variables.pow_imp[t+k],
                     'exp': self.variables.pow_exp[t+k],
                     's_k': self.variables.s_k,
                     'r_k': np.linalg.norm(self.variables.r_k),
                     'r_k[0]': real_rk[k],
                     'r_k[1]': real_rk[96+k],
                     'r_k[2]': real_rk[192+k]
                     }
                    aa = {'alfa[%s]'%(i): self.variables.alfa_k[t+k,i] for i in range(self.params.num_pros)}
                    cc = {'load[%s]'%(i): self.variables.l_k[t+k,i] for i in range(self.params.num_pros)}
                    dd = {'pow[%s]'%(i): self.variables.p_k[t+k,i] for i in range(self.params.num_pros)}
                    ee = {'price[%s]'%(i): pp[i] for i in range(self.params.num_pros)}
                    ff = {'price_imp': self.variables.price_IE[t+k,0],
                          'price_exp': self.variables.price_IE[t+k,1],
                          'price_comm': self.variables.price_comm[t+k]}
                    gg = {'q[%s]'%(i): self.variables.q_k[t+k,i] for i in range(self.params.num_pros)}
                    ii = {'beta_k[%s]'%(i): self.variables.beta_k[t+k,i] for i in range(self.params.num_pros)}
                    res.append(d)
                    res2.append(aa)
                    res4.append(cc)
                    res5.append(dd)
                    res6.append(ee)
                    res7.append(ff)
                    res8.append(gg)
                    res10.append(ii)

            print('Normal - Timestamp %g' % (t), '  Flag ', flag)
                
            fl.append(flag)
            res = pd.DataFrame(res)
            res2 = pd.DataFrame(res2)
            res4 = pd.DataFrame(res4)
            res5 = pd.DataFrame(res5)
            res6 = pd.DataFrame(res6)
            res7 = pd.DataFrame(res7)
            res8 = pd.DataFrame(res8)
            res9 = pd.DataFrame(res9)
            res10 = pd.DataFrame(res10)
            resul.append(pd.concat([res,res2,res10,res4,res5,res8], axis=1))
            resul2.append(pd.concat([res7,res6], axis=1))
            
        self.results = resul
        self.prices = resul2
        self.flag = fl

    def clean(self):
        sys.stdout = self.stdold
        del self.model
        [self.submodels_pros.get(sm).clean() for sm in self.submodels_pros.keys()]
        
    def balancing(self):
        return