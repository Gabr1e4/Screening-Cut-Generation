# import packages
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from mosek.fusion import *
import mosek.fusion.pythonic
from source import utils,env,main
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot


# ===== Test function for Sparse Ridge Regression =====
def syn_ins_test(progress_log = False,test_alg = ['SCG'],**kwargs):
    # Solver settings
    time_lim = kwargs['time_lim']
    ite_lim = kwargs['ite_lim']
    gap_lim = kwargs['gap_lim']

    # Data SCGeration setting
    num = kwargs['num']
    dim = kwargs['dim']
    rho = kwargs['rho']         
    sparsity_ = kwargs['sparsity']   
    SNR_ = kwargs['SNR']    
    gamma_ = kwargs['gamma_']

    # SCGerate single instance
    (X,Y,beta_truth) = env.sparse_generation(num = num,
                                  dim = dim,
                                  sparsity=sparsity_,
                                  rho = rho,
                                  SNR = SNR_,
                                  random_seed = None)
                    
    procedure = main.ridge_screen(
        covariate=X,response=Y,gamma=gamma_,sparsity = sparsity_)
    
    # choose the specified solver to solve conic relaxation
    solver = kwargs['solver']
    
    # dict to store time
    time_log = {}

    # dict to store cuts
    cut_log = {}

    # dict to store attributes
    att_log = {}

    # ===== solve the baseline SSR model =====

    current_time = time.time()  
    procedure.get_feas_beta_ub(relax_solver = solver) 

    # pre-solving time  
    S_SSR,Z_SSR = procedure.safe_screen()
    find_cuts_time = time.time() - current_time

    # solving time
    current_sol_time = time.time()
    SSR_model = utils.ridge_train(X = X,Y = Y,
                          sparsity=sparsity_,gamma = gamma_,
                          Outputlog=0,mip_gap = gap_lim,
                          S_set = S_SSR,Z_set = Z_SSR,
                          max_ite = ite_lim,
                          max_min = time_lim)
    end_sol_time = time.time()

    # if the model is solved to optimal
    if SSR_model.Status == 2:
        time_lim = end_sol_time - current_sol_time
    
    # if the model is not solved to optimal
    else:
        gap_lim = SSR_model.MIPGap - 1e-3

    # store time 
    time_log['SSR'] = end_sol_time - current_time
    time_log['SSR find cuts'] = find_cuts_time

    # store cuts
    cut_log['SSR_support'] = procedure.support_index
    cut_log['SSR_zero'] = procedure.zero_index

    # store attribute
    att_log['SSR obj'] = SSR_model.getObjective().getValue()
    att_log['SSR MIPGap'] = SSR_model.MIPGAP
    att_log['SSR status'] = SSR_model.Status

    # ===== solve original problem =====
    if 'Org' in test_alg:
        current_time = time.time()
        Org_model = utils.ridge_train(X = X,Y = Y,
                              sparsity=sparsity_,gamma = gamma_,
                              Outputlog=0,mip_gap = gap_lim,
                              S_set = None,Z_set = None,
                              max_ite = ite_lim,
                              max_min = time_lim)
        end_sol_time = time.time()

        # store time
        time_log['Org'] = end_sol_time - current_time

        # store attribute
        att_log['Org obj'] = Org_model.getObjective().getValue()
        att_log['Org MIPGap'] = Org_model.MIPGAP
        att_log['Org status'] = Org_model.Status

    # ===== solve with SCG =====
    if 'SCG' in test_alg:
        inc_max_num,exc_max_num = kwargs['inc_max_num'],kwargs['exc_max_num']
        inc_max_len,exc_max_len = kwargs['inc_max_len'],kwargs['exc_max_len']
        
        current_time = time.time()
        procedure.get_feas_beta_ub(relax_solver = solver)

        # pre-solving time
        S_inc,Z_inc = procedure.inclusive_cuts(max_num = inc_max_num,
                                               max_len = inc_max_len)
        S_exc,Z_exc = procedure.exclusive_cuts(max_num = exc_max_num,
                                               max_len = exc_max_len)
        find_cuts_time = time.time() - current_time
        
        # solving time
        current_sol_time = time.time()
        SCG_model = utils.ridge_train(X = X,Y = Y,
                                sparsity=sparsity_,gamma = gamma_,
                                Outputlog=0,mip_gap = gap_lim,
                                S_set = S_inc+S_exc,Z_set = Z_inc+Z_exc,
                                max_ite = ite_lim,
                                max_min = time_lim)
        end_sol_time = time.time()

        # store time 
        time_log['SCG'] = end_sol_time - current_time
        time_log['SCG find cuts'] = find_cuts_time
    
        # store cuts
        cut_log['SCG_inclusive'] = Z_inc
        cut_log['SCG_exclusive'] = S_exc
    
        # store attribute
        att_log['SCG obj'] = SCG_model.getObjective().getValue()
        att_log['SCG MIPGap'] = SCG_model.MIPGAP
        att_log['SCG status'] = SCG_model.Status

    if progress_log:
        print('time log:',time_log)
        print('attribute log:',att_log)

    return time_log,cut_log,att_log,X,Y


def real_ins_test(X,Y,progress_log = False,test_alg = ['SCG'],**kwargs):
    # Solver settings
    time_lim = kwargs['time_lim']
    ite_lim = kwargs['ite_lim']
    gap_lim = kwargs['gap_lim']

    # Instance setting
    num = X.shape[0]
    dim = X.shape[1]      
    sparsity_ = kwargs['sparsity']     
    gamma_ = kwargs['gamma_']
    
    # choose the specified solver to solve conic relaxation
    solver = kwargs['solver']
    
    # dict to store time
    time_log = {}

    # dict to store cuts
    cut_log = {}

    # dict to store attributes
    att_log = {}

    procedure = main.ridge_screen(
        covariate=X,response=Y,gamma=gamma_,sparsity = sparsity_)

    # ===== solve the baseline SSR model =====

    current_time = time.time()  
    procedure.get_feas_beta_ub(relax_solver = solver) 

    # find cuts time  
    S_SSR,Z_SSR = procedure.safe_screen()
    find_cuts_time = time.time() - current_time

    # solver time
    current_sol_time = time.time()
    SSR_model = utils.ridge_train(X = X,Y = Y,
                          sparsity=sparsity_,gamma = gamma_,
                          Outputlog=0,mip_gap = gap_lim,
                          S_set = S_SSR,Z_set = Z_SSR,
                          max_ite = ite_lim,
                          max_min = time_lim)
    end_sol_time = time.time()

    # if the model is solved to optimal
    if SSR_model.Status == 2:
        time_lim = end_sol_time - current_sol_time
    
    # if the model is not solved to optimal
    else:
        gap_lim = SSR_model.MIPGap - 1e-3

    # store time 
    time_log['SSR'] = end_sol_time - current_time
    time_log['SSR find cuts'] = find_cuts_time

    # store cuts
    cut_log['SSR_support'] = procedure.support_index
    cut_log['SSR_zero'] = procedure.zero_index

    # store attribute
    att_log['SSR obj'] = SSR_model.getObjective().getValue()
    att_log['SSR MIPGap'] = SSR_model.MIPGAP
    att_log['SSR status'] = SSR_model.Status

    # ===== solve original problem =====
    if 'Org' in test_alg:
        
        current_time = time.time()
        Org_model = utils.ridge_train(X = X,Y = Y,
                              sparsity=sparsity_,gamma = gamma_,
                              Outputlog=0,mip_gap = gap_lim,
                              S_set = None,Z_set = None,
                              max_ite = ite_lim,
                              max_min = time_lim)
        end_sol_time = time.time()

        # store time
        time_log['Org'] = end_sol_time - current_time

        # store attribute
        att_log['Org obj'] = Org_model.getObjective().getValue()
        att_log['Org MIPGap'] = Org_model.MIPGAP
        att_log['Org status'] = Org_model.Status

    # ===== solve with SCG =====
    if 'SCG' in test_alg:
        inc_max_num,exc_max_num = kwargs['inc_max_num'],kwargs['exc_max_num']
        inc_max_len,exc_max_len = kwargs['inc_max_len'],kwargs['exc_max_len']
        
        current_time = time.time()
        procedure.get_feas_beta_ub(relax_solver = solver)

        # pre-solving time
        S_inc,Z_inc = procedure.inclusive_cuts(max_num = inc_max_num,
                                               max_len = inc_max_len)
        S_exc,Z_exc = procedure.exclusive_cuts(max_num = exc_max_num,
                                               max_len = exc_max_len)
        
        find_cuts_time = time.time() - current_time
        
        # solving time
        current_sol_time = time.time()
        SCG_model = utils.ridge_train(X = X,Y = Y,
                                sparsity=sparsity_,gamma = gamma_,
                                Outputlog=0,mip_gap = gap_lim,
                                S_set = S_inc+S_exc,Z_set = Z_inc+Z_exc,
                                max_ite = ite_lim,
                                max_min = time_lim)
        end_sol_time = time.time()

        # store time 
        time_log['SCG'] = end_sol_time - current_time
        time_log['SCG find cuts'] = find_cuts_time
    
        # store cuts
        cut_log['SCG_inclusive'] = Z_inc
        cut_log['SCG_exclusive'] = S_exc
    
        # store attribute
        att_log['SCG obj'] = SCG_model.getObjective().getValue()
        att_log['SCG MIPGap'] = SCG_model.MIPGAP
        att_log['SCG status'] = SCG_model.Status

    if progress_log:
        print('time log:',time_log)
        print('attribute log:',att_log)
        # opt_z = [var.X for var in SSR_model.getVars() if 'z' in var.VarName]
        # print('optimal support:', [i for i in range(len(opt_z)) if opt_z[i] == 1])

    return time_log,cut_log,att_log



# ====== Script code ======
# def time_limit_sol_instance(progress_log = False,method = 'SSR',**kwargs):
#     # Stopping time
#     total_time = kwargs['total_time']
    
#     # Solver settings
#     time_lim = kwargs['time_lim']
#     ite_lim = kwargs['ite_lim']
#     gap_lim = kwargs['gap_lim']

#     # Data SCGeration setting
#     num = kwargs['num']
#     dim = kwargs['dim']
#     rho = kwargs['rho']         
#     sparsity_ = kwargs['sparsity']   
#     SNR_ = kwargs['SNR']    
#     gamma_i = kwargs['gamma_i']

#     start_time = time.time()
#     solved_instance = 0
#     while True:
#         # Total time limit
#         current_time = time.time()
#         if current_time - start_time > total_time:
#             break
        
#         # SCGerate instances
#         (X,Y,beta_truth) = env.sparse_SCGeration(num = num,
#                                   dim = dim,
#                                   sparsity=sparsity_,
#                                   rho = rho,
#                                   SNR = SNR_,
#                                   random_seed = None)


#         gamma_ = 1/env.SCGerate_gamma(X = X,sparsity=sparsity_,i = gamma_i)      
#         procedure = main.ridge_screen(
#         covariate=X,response=Y,gamma=gamma_,sparsity = sparsity_)


#         if method == 'SSR':
#             S_SSR,Z_SSR = procedure.safe_screen()
#             SSR_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = S_SSR,Z_set = Z_SSR,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
#             if SSR_model.Status == 2:
#                 solved_instance +=1

#         elif method == 'Org':
#             Org_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = None,Z_set = None,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
#             if Org_model.Status == 2:
#                 solved_instance +=1

#         elif method == 'SCG':
#             inc_max_num,exc_max_num = kwargs['inc_max_num'],kwargs['exc_max_num']
#             inc_max_len,exc_max_len = kwargs['inc_max_len'],kwargs['exc_max_len']
#             tail_len_,gap_len_ = kwargs['tail_len'],kwargs['gap_len']
            
#             procedure.get_int_gap()
#             S_inc,Z_inc = procedure.inclusive_cuts(max_num = inc_max_num,
#                                                    max_len = inc_max_len,
#                                                    tail_len = tail_len_)
#             S_exc,Z_exc = procedure.exclusive_cuts(max_num = exc_max_num,
#                                                    max_len = exc_max_len,
#                                                    gap_len = gap_len_)
#             SCG_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = S_inc+S_exc,Z_set = Z_inc+Z_exc,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
#             if SCG_model.Status == 2:
#                 solved_instance +=1

#         if progress_log:
#             print('instances solved:',solved_instance)

#     return solved_instance


# def time_limit_opt_gap(progress_log = False,method = 'SSR',**kwargs):
#     # Stopping time
#     total_time = kwargs['total_time']
    
#     # Solver settings
#     time_lim = kwargs['time_lim']
#     ite_lim = kwargs['ite_lim']
#     gap_lim = kwargs['gap_lim']

#     # Data SCGeration setting
#     num = kwargs['num']
#     dim = kwargs['dim']
#     rho = kwargs['rho']         
#     sparsity_ = kwargs['sparsity']   
#     SNR_ = kwargs['SNR']    
#     gamma_i = kwargs['gamma_i']

#     start_time = time.time()
#     num_instance = 0
#     opt_gap = 0
#     while True:
#         # Total time limit
#         current_time = time.time()
#         if current_time - start_time > total_time:
#             break
        
#         # SCGerate instances
#         (X,Y,beta_truth) = env.sparse_SCGeration(num = num,
#                                   dim = dim,
#                                   sparsity=sparsity_,
#                                   rho = rho,
#                                   SNR = SNR_,
#                                   random_seed = None)


#         gamma_ = 1/env.SCGerate_gamma(X = X,sparsity=sparsity_,i = gamma_i)      
#         procedure = main.ridge_screen(
#         covariate=X,response=Y,gamma=gamma_,sparsity = sparsity_)


#         if method == 'SSR':
#             S_SSR,Z_SSR = procedure.safe_screen()
#             SSR_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = S_SSR,Z_set = Z_SSR,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
            
#             num_instance +=1
#             opt_gap += SSR_model.MIPGap

#         elif method == 'Org':
#             Org_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = None,Z_set = None,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
            
#             num_instance +=1
#             opt_gap += Org_model.MIPGap

#         elif method == 'SCG':
#             inc_max_num,exc_max_num = kwargs['inc_max_num'],kwargs['exc_max_num']
#             inc_max_len,exc_max_len = kwargs['inc_max_len'],kwargs['exc_max_len']
#             tail_len_,gap_len_ = kwargs['tail_len'],kwargs['gap_len']
            
#             procedure.get_int_gap()
#             S_inc,Z_inc = procedure.inclusive_cuts(max_num = inc_max_num,
#                                                    max_len = inc_max_len,
#                                                    tail_len = tail_len_)
#             S_exc,Z_exc = procedure.exclusive_cuts(max_num = exc_max_num,
#                                                    max_len = exc_max_len,
#                                                    gap_len = gap_len_)
#             SCG_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = S_inc+S_exc,Z_set = Z_inc+Z_exc,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
            
#             num_instance +=1
#             opt_gap += SCG_model.MIPGap

#         if progress_log:
#             print('average optimality gap:',opt_gap/num_instance)

#     return opt_gap/num_instance


# def time_limit_sol_time(progress_log = False,method = 'SSR',**kwargs):
#     # Stopping time
#     total_time = kwargs['total_time']
    
#     # Solver settings
#     time_lim = kwargs['time_lim']
#     ite_lim = kwargs['ite_lim']
#     gap_lim = kwargs['gap_lim']

#     # Data SCGeration setting
#     num = kwargs['num']
#     dim = kwargs['dim']
#     rho = kwargs['rho']         
#     sparsity_ = kwargs['sparsity']   
#     SNR_ = kwargs['SNR']    
#     gamma_ = kwargs['gamma_']

#     start_time = time.time()
#     num_instance = 0
#     total_sol_time = 0
#     while True:
#         # Total time limit
#         current_time = time.time()
#         if current_time - start_time > total_time:
#             break
        
#         # SCGerate instances
#         (X,Y,beta_truth) = env.sparse_SCGeration(num = num,
#                                   dim = dim,
#                                   sparsity=sparsity_,
#                                   rho = rho,
#                                   SNR = SNR_,
#                                   random_seed = None)


#         # gamma_ = 1/env.SCGerate_gamma(X = X,sparsity=sparsity_,i = gamma_i)      
#         procedure = main.ridge_screen(
#         covariate=X,response=Y,gamma=gamma_,sparsity = sparsity_)


#         if method == 'SSR':
#             current_sol_time = time.time()
#             S_SSR,Z_SSR = procedure.safe_screen()
#             SSR_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = S_SSR,Z_set = Z_SSR,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
#             end_sol_time = time.time()
#             num_instance +=1
#             total_sol_time += (end_sol_time - current_sol_time)

#         elif method == 'Org':
#             current_sol_time = time.time()
#             Org_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = None,Z_set = None,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
#             end_sol_time = time.time()
#             num_instance +=1
#             total_sol_time += (end_sol_time - current_sol_time)

#         elif method == 'SCG':
#             inc_max_num,exc_max_num = kwargs['inc_max_num'],kwargs['exc_max_num']
#             inc_max_len,exc_max_len = kwargs['inc_max_len'],kwargs['exc_max_len']
#             tail_len_,gap_len_ = kwargs['tail_len'],kwargs['gap_len']
            
#             current_sol_time = time.time()
#             procedure.get_int_gap()
#             S_inc,Z_inc = procedure.inclusive_cuts(max_num = inc_max_num,
#                                                    max_len = inc_max_len,
#                                                    tail_len = tail_len_)
#             S_exc,Z_exc = procedure.exclusive_cuts(max_num = exc_max_num,
#                                                    max_len = exc_max_len,
#                                                    gap_len = gap_len_)
#             SCG_model = utils.ridge_train(X = X,Y = Y,
#                                   sparsity=sparsity_,gamma = gamma_,
#                                   Outputlog=0,mip_gap = gap_lim,
#                                   S_set = S_inc+S_exc,Z_set = Z_inc+Z_exc,
#                                   max_ite = ite_lim,
#                                   max_min = time_lim)
#             end_sol_time = time.time()
#             num_instance +=1
#             total_sol_time += (end_sol_time - current_sol_time)

#         if progress_log:
#             print('instances:',num_instance)
#             print('average solution time:',total_sol_time/num_instance)

#     return total_sol_time/num_instance