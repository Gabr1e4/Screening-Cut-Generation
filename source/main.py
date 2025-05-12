# import packages
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from mosek.fusion import *
import mosek.fusion.pythonic
from source import utils,env


# ===== Screening implementation for Ridge Regression =====
class ridge_screen():
    def __init__( 
        self,
        covariate: np.ndarray,
        response: np.ndarray,
        sparsity: int,
        gamma:float) -> None:
        """Implement the screening method for the ridge regression model.
        Args:
            covariate (np.ndarray): the covariate matrix, X.
            response (np.ndarray): the response vector, Y.
            sparsity (int): the sparsity level.
            gamma (float): the coefficient for l2 penalty term
        """
        self.covariate = covariate
        self.response = response
        self.num = covariate.shape[0]
        self.dim = covariate.shape[1]
        self.sparsity = sparsity
        self.gamma = gamma
        self.support_index = []
        self.zero_index = []
    
    def get_feas_beta_ub(self,relax_solver = 'gurobi'):
        # solve the primal relaxation problem

        # === Mosek solver ===
        if relax_solver == 'mosek':
            self.relax_model = utils.ridge_relax_mosek(
                    X = self.covariate,
                    Y = self.response,
                    sparsity = self.sparsity,
                    gamma = self.gamma)

            self.feas_beta = np.array(list(self.relax_model.getVariable('beta').level()))
            self.relax_value = self.relax_model.primalObjValue()
            
            relax_z = list(self.relax_model.getVariable('z').level())
            subs_z = list(self.relax_model.getVariable('z').level())

        # === Gurobi solver ===
        elif relax_solver == 'gurobi':
            self.relax_model = utils.ridge_relax(
                    X = self.covariate,
                    Y = self.response,
                    sparsity = self.sparsity,
                    gamma = self.gamma)
            
            self.feas_beta = np.array([var.X for var in self.relax_model.getVars() if 'beta' in var.VarName])
            self.relax_value = self.relax_model.getObjective().getValue()
            
            relax_z = [var.X for var in self.relax_model.getVars() if 'z' in var.VarName]
            subs_z = [var.X for var in self.relax_model.getVars() if 'z' in var.VarName]
        
        # find the rounding z
        round_z = np.zeros(self.dim)
        round_index = []
        for _ in range(self.sparsity):
            max_value = max(subs_z)
            max_index = relax_z.index(max_value)
            round_index.append(max_index)
            subs_z.remove(max_value)
            
        round_z[round_index] = 1    
        diag_z = np.diag(round_z)
        
        # find the upper bound
        self.upper_bound_model = utils.ridge_train(
            X = self.covariate@diag_z, 
            Y = self.response, 
            gamma = self.gamma,
            sparsity = 0)
        
        self.upper_bound = self.upper_bound_model.getObjective().getValue()

    
    def recursive_iteration(self,weight_limit,item_list_sort,index_set,select_num,lst_rst = [],lst_tmp = []):
        # The function serves as the recursive iteration function to find T_s and T'_s
        for i in range(len(index_set) - select_num+1):
            x_idx = index_set[i]
            item_weight = item_list_sort[x_idx]
            rest_weight_minimum = weight_limit - item_weight \
                                - sum(item_list_sort[index_set[-1] - (select_num-2):index_set[-1]+1])
            if rest_weight_minimum < 0:
                continue
            else:
                lst_tmp.append(x_idx)
                if select_num == 1:
                    lst_rst.append([*lst_tmp])
                else:
                    self.recursive_iteration(weight_limit - item_weight,item_list_sort,
                                index_set[i+1:],select_num - 1,lst_rst,lst_tmp)
                lst_tmp.pop()
        return lst_rst
    
    
    def find_order(self,x,current_list,map_list):
        # The function serves as the re-ordering function
        ans = []
        for index in x:
            item_weight = current_list[index]
            target_list = [i for i in range(len(map_list)) if map_list[i] == item_weight]

            # has no same value
            if len(target_list) == 1:
                ans.append(target_list[0])

            # has same value
            else:
                match_list = [i for i in range(len(current_list)) if current_list[i] == item_weight]
                ans.append(target_list[match_list.index(index)])

        return ans


    def safe_screen(self):
        S_set,Z_set = [],[]

        # recover p
        p = 2/(self.num*self.gamma)*self.covariate.T@(self.response - self.covariate@self.feas_beta)

        p_square = p*p
        p_square_sort = list(p_square)
        p_square_sort.sort(reverse = True)
        p_sqaure_k = p_square_sort[self.sparsity-1]
        p_square_k_plus_one = p_square_sort[self.sparsity]

        weight_limit = - (4/self.gamma) * (self.upper_bound - self.relax_value) + np.sum(p_square_sort[:self.sparsity])

        # start safe-screen
        for index in range(self.dim):
            p_square_j = p_square[index]

            if (p_square_j <= p_square_k_plus_one) and (p_square_j + np.sum(p_square_sort[0:self.sparsity-1]) < weight_limit):
                self.zero_index.append(index)
                S_set.append([index]),Z_set.append([])
            
            # Never satisfied
            # elif (p_square_j >= p_sqaure_k) and (np.sum(p_square_sort[0:self.sparsity]) < weight_limit):
            #     self.zero_index.append(index)
            #     S_set.append([index]),Z_set.append([])

            if (p_square_j >= p_sqaure_k) and (np.sum(p_square_sort[0:self.sparsity+1]) - p_square_j < weight_limit):
                self.support_index.append(index)
                S_set.append([]),Z_set.append([index])
                
            # Never satisfied
            # elif (p_square_j <= p_square_k_plus_one) and (np.sum(p_square_sort[0:self.sparsity]) < weight_limit):
            #     self.support_index.append(index)
            #     S_set.append([]),Z_set.append([index])
                 
        return S_set,Z_set

  
    def inclusive_cuts(self,max_num = 500,max_len = None):

        if max_len == None:
            max_len = self.dim - self.sparsity
        assert (max_len >= 1) and (max_len <= self.dim - self.sparsity), "max_len should be in [1,dim-sparsity]"
        
        S_set,Z_set = [],[]
        
        # recover p
        p = 2/(self.num*self.gamma)*self.covariate.T@(self.response - self.covariate@self.feas_beta)
        
        p_square = p*p
        p_square_sort = list(p_square)
        p_square_sort.sort(reverse = True)
        
        # Construct Knapsack set 
        weight_limit = (1/4)*(self.gamma)*(np.sum(p_square_sort[0:self.sparsity])) - (self.upper_bound - self.relax_value)
        item_weight = (1/4)*(self.gamma)*p_square
        item_weight_sort = (1/4)*(self.gamma)*np.array(p_square_sort)
        
        # initialize searching index s
        last_s = None
        for i in range(1,max_len + 1):
            if weight_limit > sum(item_weight_sort[i:i+self.sparsity]):
                last_s = i + self.sparsity - 1
                break
        if last_s == None:
            return [[]],[[]]
        
        stop_flag = False
        while not stop_flag:
            x_set = self.recursive_iteration(
                                    weight_limit - item_weight_sort[last_s],
                                    item_weight_sort,
                                    [i for i in range(last_s)],
                                    self.sparsity - 1,
                                    [],[last_s])
            
            if last_s == self.sparsity + max_len - 1:
                stop_flag = True

            for x in x_set:
                S,Z = [],[i for i in range(last_s+1) if (i not in x)]
                Z = self.find_order(Z,item_weight_sort,item_weight)
                if (Z == []) or (True in [set(Z_set[i]).issubset(set(Z)) for i in range(len(Z_set))]):
                    continue
                else:
                    S_set.append(S)
                    Z_set.append(Z)
                    max_num -= 1
                    if max_num == 0:
                        stop_flag = True
                        break
                    
            last_s += 1
            
        return S_set,Z_set


    def exclusive_cuts(self,max_num = 500,max_len = None):
        
        if max_len == None:
            max_len = self.sparsity - 1

        assert (max_len >= 1) and (max_len <= self.sparsity - 1), "max_len should be in [1,sparsity-1]"

        S_set,Z_set = [],[]
       
        # recover p
        p = 2/(self.num*self.gamma)*self.covariate.T@(self.response - self.covariate@self.feas_beta)
        
        p_square = p*p
        p_square_sort = list(p_square)
        p_square_sort.sort(reverse = True)
        
        # Construct Knapsack set
        weight_limit = (1/4)*(self.gamma)*(np.sum(p_square_sort[0:self.sparsity])) - (self.upper_bound - self.relax_value)
        item_weight = (1/4)*(self.gamma)*p_square
        item_weight_sort = (1/4)*(self.gamma)*np.array(p_square_sort)

        # Initialize max_select
        max_select = None
        for select_num in range(self.sparsity - max_len,self.sparsity):
            min_weight = sum(item_weight_sort[:select_num]) + \
                    sum(item_weight_sort[self.dim-self.sparsity + select_num:self.dim])
            
            if (min_weight < weight_limit):
                max_select = select_num
        
        if max_select == None:
            return [[]],[[]]

        # if we can fix some zeros, then we can reduce the search region
        elif max_select == self.sparsity - 1:
            fix_zero = [[i] for i in range(self.sparsity,self.dim) if sum(item_weight_sort[:max_select]) + item_weight_sort[i] < weight_limit]
            last_index = min([index[0] for index in fix_zero]) 
            for S in fix_zero:
                S = self.find_order(S,item_weight_sort,item_weight)
                S_set.append(S)
                Z_set.append([])
                max_num -= 1

                if max_num == 0:
                    return S_set,Z_set

            if max_len == 1:
                return S_set,Z_set

            max_select -= 1

        else:
            last_index = self.dim

        
        stop_flag = False
        while not stop_flag:
            x_set = self.recursive_iteration(weight_limit - sum(item_weight_sort[:max_select]),
                                        item_weight_sort,
                                        [i for i in range(max_select+1,last_index)],
                                        self.sparsity - max_select,
                                        [],[])
            
            
            if max_select == self.sparsity - max_len:
                stop_flag = True
            
            for S in x_set:
                S = self.find_order(S,item_weight_sort,item_weight)
                
                if (S == []) or (True in [set(S_set[i]).issubset(set(S)) for i in range(len(S_set))]):
                    continue
                else:
                    S_set.append(S)
                    Z_set.append([])
                    max_num -= 1

                    if max_num == 0:
                        stop_flag = True
                        break

            max_select -=1
            
        return S_set,Z_set

