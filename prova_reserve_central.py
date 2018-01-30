# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:01:15 2018

@author: Chiara
"""
import numpy as np
import gurobipy as gb

Cp = np.array([10,30])
CR = np.array([0,25])
uncertainty = 0.05
Pmin = np.array([0,0])
Pmax = np.array([100,100])
Voll = 1000
DEM = 130
RES = 20
Smax = 20

n = len(Pmax)

p_sol = np.zeros(2)
R_sol = np.zeros(2)
r_sol = np.zeros(2)
lShed_sol = 0.0
priceDEM_sol = 0.0
priceRES_sol = 0.0

res_m = gb.Model("qp")
res_m.setParam( 'OutputFlag', False )
# Create variables
p = np.array([res_m.addVar() for i in range(n)])
r = np.array([res_m.addVar() for i in range(n)]) 
R = np.array([res_m.addVar() for i in range(n)])
lShed = res_m.addVar()
    
res_m.update()
# Set objective:
obj = sum(Cp*p) + sum(CR*R) + uncertainty*(sum(Cp*r) + Voll*lShed)
res_m.setObjective(obj)
    
# Add constraint
res_m.addConstr(sum(p[:]) == DEM, name="dem")
res_m.addConstr(sum(r[:]) + lShed == RES, name="res")
for i in range(n): 
    res_m.addConstr(r[i] - R[i] <= 0, name="r_limit[%s]"%(i))
    res_m.addConstr(p[i] + R[i] <= Pmax[i], name="p_limit[%s]"%(i)) 

res_m.update()    
res_m.optimize()

for i in range(n):
    p_sol[i] = p[i].x
    r_sol[i] = r[i].x
    R_sol[i] = R[i].x
priceDEM_sol = res_m.getConstrByName("dem").Pi
priceRES_sol = res_m.getConstrByName("res").Pi

del res_m

