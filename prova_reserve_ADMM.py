# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:24:32 2018

@author: Chiara
"""

import numpy as np
import gurobipy as gb

Cp = np.array([10,30])
CR = np.array([0,25])
uncertainty = 0.05
Pmax = np.array([100,100])
Voll = 1000
DEM = 130
RES = 20
Smax = 20

n = len(Pmax)

#initialize variables
p0_k = 0
R0_k = 0
r0_k = 0
p1_k = 0
R1_k = 0
r1_k = 0
rho = 0.01
lShed_k = 0
lambda_dem = 0
lambda_res = 0
count =0
r_dem = 1
r_res = 1

while ((abs(r_dem) > 1e-5)|(abs(r_res) > 1e-5)):
    count +=1
    # PRODUCER 0
    res_m = gb.Model("qp")
    res_m.setParam( 'OutputFlag', False )    
    # Create variables
    p0 = res_m.addVar()
    r0 = res_m.addVar()
    R0 = res_m.addVar()
    res_m.update()   
    # Set objective:
    obj = Cp[0]*p0 + CR[0]*R0 + uncertainty*Cp[0]*r0 + \
          lambda_dem*p0 + lambda_res*r0 + \
          rho/2*((p0 + p1_k - DEM)*(p0 + p1_k - DEM) + (r0 + r1_k + lShed_k - RES)*(r0 + r1_k + lShed_k - RES))
    res_m.setObjective(obj)    
    # Add constraints
    res_m.addConstr(r0 - R0 <= 0, name="r_limit0")
    res_m.addConstr(p0 + R0 <= Pmax[0], name="p_limit0") 
    res_m.update()    
    res_m.optimize()
    # save results
    p0_k = p0.x
    r0_k = r0.x
    R0_k = R0.x

    # PRODUCER 1
    res_m = gb.Model("qp")
    res_m.setParam( 'OutputFlag', False )    
    # Create variables
    p1 = res_m.addVar()
    r1 = res_m.addVar()
    R1 = res_m.addVar()
    res_m.update()    
    # Set objective:
    obj = Cp[1]*p1 + CR[1]*R1 + uncertainty*Cp[1]*r1 + \
          lambda_dem*p1 + lambda_res*r1 + \
          rho/2*((p0_k + p1 - DEM)*(p0_k + p1 - DEM) + (r0_k + r1 + lShed_k - RES)*(r0_k + r1 + lShed_k - RES))
    res_m.setObjective(obj)    
    # Add constraints
    res_m.addConstr(r1 - R1 <= 0, name="r_limit1")
    res_m.addConstr(p1 + R1 <= Pmax[1], name="p_limit1") 
    res_m.update()    
    res_m.optimize()
    # save results
    p1_k = p1.x
    r1_k = r1.x
    R1_k = R1.x

    # MAIN
    res_m = gb.Model("qp")
    res_m.setParam( 'OutputFlag', False )    
    # Create variables
    lShed = res_m.addVar()
    res_m.update()    
    # Set objective:
    obj = uncertainty*Voll*lShed + lambda_res*lShed + \
          rho/2*((r0_k + r1_k + lShed - RES)*(r0_k + r1_k + lShed - RES))
    res_m.setObjective(obj)   
    # Add constraints
    res_m.addConstr(lShed - Smax <= 0, name="shed_limit")
    res_m.update()    
    res_m.optimize()
    # save results
    lShed_k = lShed.x

    # UPDATE dual variables
    lambda_dem = lambda_dem + rho*(p0_k + p1_k - DEM)
    lambda_res = lambda_res + rho*(r0_k + r1_k + lShed_k - RES)
    
    # calculate RESIDUALS
    r_dem = p0_k + p1_k - DEM
    r_res = r0_k + r1_k + lShed_k - RES