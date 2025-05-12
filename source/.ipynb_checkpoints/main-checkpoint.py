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

        weight_limit = (4/self.gamma)*((1/self.num)*np.linalg.norm(self.covariate@self.feas_beta - self.response,2)**2 + \
                                                        self.gamma * p @ self.feas_beta - self.upper_bound)
        # weight_limit = - (4/self.gamma) * (self.upper_bound - self.relax_value) + np.sum(p_square_sort[:self.sparsity])

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
        # weight_limit = (1/4)*(self.gamma)*(np.sum(c_square_sort[0:self.sparsity])) - (self.upper_bound - self.relax_value)
        weight_limit = (1/self.num) * (self.covariate@self.feas_beta - self.response).T@(self.covariate@self.feas_beta - self.response) \
                                + self.gamma * p@self.feas_beta - self.upper_bound
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
        # weight_limit = (1/4)*(self.gamma)*(np.sum(c_square_sort[0:self.sparsity])) - (self.upper_bound - self.relax_value)
        weight_limit = (1/self.num) * (self.covariate@self.feas_beta - self.response).T@(self.covariate@self.feas_beta - self.response) \
                        + self.gamma * p@self.feas_beta - self.upper_bound
        item_weight = (1/4)*(self.gamma)*p_square
        item_weight_sort = (1/4)*(self.gamma)*np.array(p_square_sort)

        # Initialize searching index s
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

        # (Find maximal clique to iterate all exclusive cuts with len = 2, or max_select = k-2)
        
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


# ===== Script function below =====
# def get_upper_bound(self):
#     # First solve the relaxed primal problem
#     self.solve_primal_relax()
#     # print('relax model status:',self.relax_model.Status)
    
#     relax_z = [var.X for var in self.relax_model.getVars() if 'z' in var.VarName]
#     subs_z = [var.X for var in self.relax_model.getVars() if 'z' in var.VarName]
#     # relax_z = list(self.relax_model.getVariable('z').level())
#     # subs_z = list(self.relax_model.getVariable('z').level())
    
#     # find the rounding z
#     round_z = np.zeros(self.dim)
#     round_index = []
#     for _ in range(self.sparsity):
#         max_value = max(subs_z)
#         max_index = relax_z.index(max_value)
#         round_index.append(max_index)
#         subs_z.remove(max_value)
        
#     round_z[round_index] = 1    
#     diag_z = np.diag(round_z)
    
#     # find the upper bound
#     self.upper_bound_model = utils.ridge_train(
#         X = self.covariate@diag_z, 
#         Y = self.response, 
#         gamma = self.gamma,
#         sparsity = 0)
    
#     # print('upper bound model status:',self.upper_bound_model.Status)
#     upper_bound_obj = self.upper_bound_model.getObjective()
#     self.upper_bound = upper_bound_obj.getValue()
    

# def get_int_gap(self):
#     self.get_upper_bound()
#     self.int_gap = self.upper_bound - self.relax_model.getObjective().getValue()

    
# def check_feasible(self,x,item_list,weight_limit):
#     if (len(x) <= self.sparsity) and (sum(item_list[x]) < weight_limit):
#         return True
#     else:
#         return False


# def find_cuts(self,x,item_list_sort,s_ub = None,total_ub = None):

#     if s_ub == None:
#         s_ub = len(item_list_sort)
        
#     if total_ub == None:
#         total_ub = len(item_list_sort)
    
#     S_set = []
#     Z_set = []
#     x_sort= sorted(x)
    
#     if len(x) < self.sparsity:
#         if ([i for i in range(len(item_list_sort)) if i not in x] <= total_ub):
#             S_set.append([])
#             Z_set.append([i for i in range(len(item_list_sort)) if i not in x])
#     else:
#         M_list = []
#         current_group = []
#         for i in range(len(x_sort)):
#             current_group.append(x_sort[i])
#             if i == len(x_sort)-1:
#                 M_list.append(current_group)
#                 break

#             else:
#                 if x_sort[i+1] > x_sort[i]+1:
#                     M_list.append(current_group)
#                     current_group = []

#         Z_list = []
#         for i in range(len(M_list)):
#             M_i = M_list[i]
#             if i == 0:
#                 Z_list.append([i for i in range(min(M_i))])
#             else:
#                 Z_list.append([i for i in range(max(M_list[i-1])+1,min(M_i))]) 

#         # Construct cuts
#         if (len(x)<= s_ub) and (len(x) <= total_ub):
#             S_set.append(x)
#             Z_set.append([])
        
#         for i in range(len(M_list)):
#             Z = sum([Z_list[j] for j in range(i+1)],[])
#             S = sum([M_list[j] for j in range(i+1,len(M_list))],[])
#             if (len(S)<=s_ub) and (len(S)+len(Z)<=total_ub):
#                 S_set.append(S)
#                 Z_set.append(Z)

#     return S_set,Z_set


#  def logic_rule_screen(self):
#     S_set,Z_set = [],[]

#     # First get the upper bound and lower bound
#     self.get_upper_bound()
#     relax_obj = self.relax_model.getObjective().getValue()

#     # get the relaxed beta
#     self.relax_beta = np.array([var.X for var in self.relax_model.getVars() if 'beta' in var.VarName])
    
#     # recover c
#     c = 2/(self.num*self.gamma)*self.covariate.T@(self.response - self.covariate@self.relax_beta)
#     c_k = np.sqrt(np.sort(c*c)[-self.sparsity])
#     c_k_plus_one = np.sqrt(np.sort(c*c)[-self.sparsity-1])
#     c_k_minus_one = np.sqrt(np.sort(c*c)[-self.sparsity+1])
#     c_k_plus_two = np.sqrt(np.sort(c*c)[-self.sparsity-2])

#     # start logic-screen
#     for i in range(self.dim):
#         for j in range(i+1,self.dim):
#             c_i = c[i]
#             c_j = c[j]

#             # Check z_i=z_j=1
#             both_out_flag = (c_i**2 <= c_k_plus_one**2) and (c_j**2 <= c_k_plus_one**2)
#             i_k_j_out_flag = (c_i**2 == c_k**2) and (c_j**2 <= c_k_plus_one**2)
#             i_in_j_out_flag = (c_i**2 >= c_k_minus_one**2) and (c_j**2 <= c_k_plus_one**2)
#             j_k_i_out_flag = (c_j**2 == c_k**2) and (c_i**2 <= c_k_plus_one**2)
#             j_in_i_out_flag = (c_j**2 >= c_k_minus_one**2) and (c_i**2 <= c_k_plus_one**2)

#             if both_out_flag ==True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_k**2 + c_k_minus_one**2 - c_i**2 - c_j**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'<=1')
#                     S_set.append([i,j])
#                     Z_set.append([])
#             elif i_k_j_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_k_minus_one**2 - c_j**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'<=1')
#                     S_set.append([i,j])
#                     Z_set.append([])
#             elif i_in_j_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_k**2 - c_j**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'<=1')
#                     S_set.append([i,j])
#                     Z_set.append([])
#             elif j_k_i_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_k_minus_one**2 - c_i**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'<=1')
#                     S_set.append([i,j])
#                     Z_set.append([])
#             elif j_in_i_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_k**2 - c_i**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'<=1')
#                     S_set.append([i,j])
#                     Z_set.append([])

#             # Check z_i=z_j=0
#             both_in_flag = (c_i**2 >= c_k**2) and (c_j**2 >= c_k**2)
#             i_in_j_k_plus_one_flag = (c_i**2 >= c_k**2) and (c_j**2 == c_k_plus_one**2)
#             i_in_j_out_flag = (c_i**2 >= c_k**2) and (c_j**2 < c_k_plus_one**2)
#             j_in_i_k_plus_one_flag = (c_j**2 >= c_k**2) and (c_i**2 == c_k_plus_one**2)
#             j_in_i_out_flag = (c_j**2 >= c_k**2) and (c_i**2 < c_k_plus_one**2)

#             if both_in_flag ==True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_i**2 + c_j**2-c_k_plus_one**2 - c_k_plus_two**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'>=1')
#                     S_set.append([])
#                     Z_set.append([i,j])
#             elif i_in_j_k_plus_one_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_i**2-c_k_plus_two**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'>=1')
#                     S_set.append([])
#                     Z_set.append([i,j])
#             elif i_in_j_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_i**2-c_k_plus_one**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'>=1')
#                     S_set.append([])
#                     Z_set.append([i,j])
#             elif j_in_i_k_plus_one_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_j**2-c_k_plus_two**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'>=1')
#                     S_set.append([])
#                     Z_set.append([i,j])
#             elif j_in_i_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_j**2-c_k_plus_one**2) > self.upper_bound:
#                     # print('z_'+str(i)+'+'+'z_'+str(j)+'>=1')
#                     S_set.append([])
#                     Z_set.append([i,j])

#             # Check z_i=0,z_j=1
#             both_in_flag = (c_i**2 >= c_k**2) and (c_j**2 >= c_k**2)
#             both_out_flag = (c_i**2 <= c_k_plus_one**2) and (c_j**2 <= c_k_plus_one**2)
#             i_in_j_out_flag = (c_i**2 >= c_k**2) and (c_j**2 <= c_k_plus_one**2)
#             j_in_i_out_flag = (c_j**2 >= c_k**2) and (c_i**2 <= c_k_plus_one**2)

#             if both_in_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_i**2 - c_k_plus_one**2) > self.upper_bound:
#                     # print('z_'+str(j)+'<='+'z_'+str(i))
#                     S_set.append([j])
#                     Z_set.append([i])
#             elif both_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_k**2 - c_j**2) > self.upper_bound:
#                     # print('z_'+str(j)+'<='+'z_'+str(i))
#                     S_set.append([j])
#                     Z_set.append([i])
#             elif i_in_j_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_i**2 - c_j**2) > self.upper_bound:
#                     # print('z_'+str(j)+'<='+'z_'+str(i))
#                     S_set.append([j])
#                     Z_set.append([i])
                    
#             # Check z_j=0,z_i=1
#             both_in_flag = (c_i**2 >= c_k**2) and (c_j**2 >= c_k**2)
#             both_out_flag = (c_i**2 <= c_k_plus_one**2) and (c_j**2 <= c_k_plus_one**2)
#             i_in_j_out_flag = (c_i**2 >= c_k**2) and (c_j**2 <= c_k_plus_one**2)
#             j_in_i_out_flag = (c_j**2 >= c_k**2) and (c_i**2 <= c_k_plus_one**2)

#             if both_in_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_j**2 - c_k_plus_one**2) > self.upper_bound:
#                     # print('z_'+str(i)+'<='+'z_'+str(j))
#                     S_set.append([i])
#                     Z_set.append([j])
#             elif both_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_k**2 - c_i**2) > self.upper_bound:
#                     # print('z_'+str(i)+'<='+'z_'+str(j))
#                     S_set.append([i])
#                     Z_set.append([j])
#             elif j_in_i_out_flag == True:
#                 if relax_obj + (1/4)*(self.gamma)*(c_j**2 - c_i**2) > self.upper_bound:
#                     # print('z_'+str(i)+'<='+'z_'+str(j))
#                     S_set.append([i])
#                     Z_set.append([j])
#     return (S_set,Z_set)