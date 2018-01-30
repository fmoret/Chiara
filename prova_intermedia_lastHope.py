# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:59:42 2018

@author: Chiara
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:48:57 2018

@author: Chiara

"""

import numpy as np
import gurobipy as gb

Cp = np.array([10,30])
Cl  = np.array([5,35])
CR_UP = np.array([0,25])
CR_DW = np.array([0,25])
uncertainty = 0.05
Pmin = np.array([0,0])
Pmax = np.array([100,100])
Lmin = np.array([-200,-200])
Lmax = np.array([0,0])

gamma_i = 22
gamma_e = 20

RES_UP = 20
RES_DW = 20

n = len(Pmax)

p_sol = np.zeros(2)
l_sol = np.zeros(2)
q_sol = np.zeros(2)
alpha_sol = np.zeros(2)
beta_sol = np.zeros(2)
R_UP_sol = np.zeros(2*n)
R_DW_sol = np.zeros(2*n)
r_UP_sol = np.zeros(2*n)
r_DW_sol = np.zeros(2*n)
priceRES_sol = 0.0

res_m = gb.Model("qp")
#res_m.setParam( 'OutputFlag', False )
# Create variables
p = np.array([res_m.addVar() for i in range(n)])
l = np.array([res_m.addVar(lb=-gb.GRB.INFINITY, ub=0) for i in range(n)])
q = np.array([res_m.addVar() for i in range(n)])
r_UP = np.array([res_m.addVar() for i in range(2*n)]) 
r_DW = np.array([res_m.addVar(lb=-gb.GRB.INFINITY, ub=0) for i in range(2*n)]) 
R_UP = np.array([res_m.addVar() for i in range(2*n)])
R_DW = np.array([res_m.addVar() for i in range(2*n)])
alpha = np.array([res_m.addVar() for i in range(n)])
beta = np.array([res_m.addVar() for i in range(n)])
q_imp = res_m.addVar()
q_exp = res_m.addVar()
    
res_m.update()
# Set objective:
obj =   sum(Cp*p) + sum(Cl*l) +\
        sum(CR_UP*R_UP[:n]) + sum(CR_DW*R_DW[n:]) +\
        uncertainty*(sum(Cp*r_UP[:n])-sum(Cl*r_UP[n:])) +\
        uncertainty*(sum(Cp*r_DW[:n])-sum(Cl*r_DW[n:])) +\
        gamma_i*q_imp - gamma_e*q_exp + 0.001*sum(q)
res_m.setObjective(obj)
    
# Add constraint
res_m.addConstr(sum(r_UP[:]) == RES_UP, name="res_up")
res_m.addConstr(sum(-r_DW[:]) == RES_DW, name="res_dw")
#res_m.addConstr(sum(p[:])-sum(l[:]) == 0, name="bal")
res_m.addConstr(sum(q[:]) == 0, name="comm")
res_m.addConstr(sum(alpha[:]) - q_imp == 0, name="imp_bal")
res_m.addConstr(sum(beta[:]) - q_exp == 0, name="exp_bal")
for i in range(n): 
    res_m.addConstr(p[i] + l[i] + q[i] + alpha[i] - beta[i] == 0, name="pros[%s]"%(i))
    res_m.addConstr(p[i] + R_UP[i] <= Pmax[i], name="R_up_limit_p[%s]"%(i)) 
    res_m.addConstr(p[i] - R_DW[i] >= Pmin[i], name="R_dw_limit_p[%s]"%(i)) 
    res_m.addConstr(l[i] + R_UP[i+n] <= Lmax[i], name="R_up_limit_l[%s]"%(i)) 
    res_m.addConstr(l[i] - R_DW[i+n] >= Lmin[i], name="R_dw_limit_l[%s]"%(i)) 
for i in range(2*n):
    res_m.addConstr(r_UP[i] - R_UP[i] <= 0, name="r_up_limit[%s]"%(i))
    res_m.addConstr(-r_DW[i] - R_DW[i] <= 0, name="r_dw_limit[%s]"%(i))

res_m.update()    
res_m.optimize()

for i in range(n):
    p_sol[i] = p[i].x
    l_sol[i] = l[i].x
    q_sol[i] = q[i].x
    alpha_sol[i] = alpha[i].x
    beta_sol[i] = beta[i].x
for i in range(2*n):
    r_UP_sol[i] = r_UP[i].x
    R_UP_sol[i] = R_UP[i].x
    r_DW_sol[i] = r_DW[i].x
    R_DW_sol[i] = R_DW[i].x
q_imp_sol = q_imp.x
q_exp_sol = q_exp.x
priceRES_UP_sol = res_m.getConstrByName("res_up").Pi
priceRES_DW_sol = res_m.getConstrByName("res_dw").Pi
del res_m

