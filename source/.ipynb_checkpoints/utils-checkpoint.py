# ----- import packages ----- 
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from mosek.fusion import *
import mosek.fusion.pythonic
import sys

# ===== Ridge Regression with l2 penalty solver function =====
def ridge_train(
    X:np.ndarray,
    Y:np.ndarray,
    sparsity:int,
    gamma:float,
    Outputlog = 0,
    S_set = None,
    Z_set = None,
    max_ite = None,
    max_min = None,
    mip_gap = 1e-2):

    
    Y = Y.flatten()
    num = X.shape[0]
    dim = X.shape[1]

    # Gurobi Solver
    m = gp.Model('ridge regression model w/wo sparsity')
    m.Params.LogToConsole = Outputlog

    # Set Stopping criterion
    if max_ite is not None:
        m.Params.IterationLimit = max_ite

    if max_min is not None:
        m.Params.TimeLimit = max_min

    m.Params.MIPGap = mip_gap

    # add variables
    beta = m.addMVar(dim,vtype = 'C',name = 'beta',lb = -GRB.INFINITY,ub = GRB.INFINITY)


    # set objective
    m.setObjective((1 / num) * (Y -X@beta)@(Y-X@beta) + gamma*beta@beta,GRB.MINIMIZE)

    # add sparsity constraint
    if sparsity > 0:
        z = m.addMVar(dim,vtype=GRB.BINARY,name = 'z')
        m.addConstr(gp.quicksum(z)<=sparsity)
        m.addConstrs((1-z[i])*beta[i] == 0 for i in range(dim))
        
        # adding cuts
        if (S_set is not None) and (Z_set is not None):
            for i in range(len(S_set)):
                S,Z = S_set[i],Z_set[i]
                if S ==[] and Z ==[]:
                    continue
                else:
                    m.addConstr(gp.quicksum(z[j] for j in S) <= len(S) - 1 + gp.quicksum(z[j] for j in Z))
    m.optimize()
    return m


def ridge_relax(
    X:np.ndarray,
    Y:np.ndarray,
    sparsity:int,
    gamma:float,
    Outputlog = 0,
    max_ite = None,
    max_min = None):

    
    Y = Y.flatten()
    num = X.shape[0]
    dim = X.shape[1]

    # Gurobi Solver
    m = gp.Model('conic relaxation of sparse ridge regression model')
    m.Params.LogToConsole = Outputlog

    # Set Stopping criterion
    if max_ite is not None:
        m.Params.IterationLimit = max_ite

    if max_min is not None:
        m.Params.TimeLimit = max_min


    # add variables
    beta = m.addMVar(dim,vtype = 'C',name = 'beta',lb = -GRB.INFINITY,ub = GRB.INFINITY)
    z = m.addMVar(dim,vtype='C',name = 'z',lb = 0.0, ub = 1.0)
    t = m.addMVar(dim,vtype='C',name = 't',lb = 0.0)

    # set objective
    m.setObjective((1 / num) * (Y -X@beta)@(Y-X@beta) + gamma*gp.quicksum(t),GRB.MINIMIZE)

    # add constraints
    m.addConstr(gp.quicksum(z)<=sparsity)
    for i in range(dim):
        m.addConstr(t[i]*z[i] >= beta[i]*beta[i])
    m.optimize()
    return m


def ridge_relax_mosek( 
    X:np.ndarray,
    Y:np.ndarray,
    sparsity:int,
    gamma:float):


    # transform the data type into float64
    X = np.asarray(X, dtype=np.float64)
    
    Y = Y.flatten()
    Y = np.asarray(Y, dtype=np.float64)
    
    num = X.shape[0]
    dim = X.shape[1]

    m = Model('conic relaxation of sparse ridge regression model')

    # add variables
    beta = m.variable('beta',dim,Domain.unbounded())
    z = m.variable('z',dim,Domain.greaterThan(0.0))

    # add auxiliary variables
    t = m.variable('t',Domain.greaterThan(0.0))
    fenchel_term = m.variable('fenchel',dim,Domain.greaterThan(0.0))

    # add notations
    res = Y - X @ beta
    second_term = gamma*Expr.sum(fenchel_term)

    # objective
    m.objective(ObjectiveSense.Minimize, t / num +second_term) 

    # add constraints
    m.constraint(Expr.vstack(t, 1/2,res), Domain.inRotatedQCone())   # t>= ||Y - X@beta||_2^2
    m.constraint(z,Domain.lessThan(1.0))
    for i in range(dim):
        m.constraint(Expr.vstack(fenchel_term[i], (1/2)*z[i],beta[i]), Domain.inRotatedQCone()) 
    m.constraint(Expr.sum(z)-sparsity,Domain.lessThan(0.0))

    # m.setLogHandler(sys.stdout)
    m.solve()
    return m


