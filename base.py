########################################################################################################################
# Robust Optimization for the Dual Sourcing Inventory Routing Problem in Disaster Relief
# Adaptive Large Neighborhood Search Algorithm
# Python>=3.8
# Author: Weibo Zheng
########################################################################################################################

import numpy as np
import math
import re
import random
import copy

##################################################################################

# ------------------------------------ Class --------------------------------------------


class Problem:
    def __init__(self, name, locations, initial_inventory_level):
        self.instance_name = name
        self.T = 8  # planning period
        self.h = 2  # unit holding cost
        self.s = 20  # unit shortage cost
        self.dm = 4  # demand mean
        self.sigma = 1   # demand standard deviation
        self.dh = 2 # demand maximum deviation
        self.kv = 60  # speed of vehicle
        self.hv = 260 # speed of helicopter
        self.f_v = 200  # vehicle unit using cost
        self.f_h = 1000 # helicopter unit using cost
        self.temperature = 30000
        self.cooling_rate = 0.995
        self.segments_capacity = 50
        self.removal_rate = 0.2
        self.reaction_factor = 0.2
        self.score_factor_1 = 10
        self.score_factor_2 = 5
        self.score_factor_3 = 2
        self.locations = locations
        self.initial_inventory_level = initial_inventory_level
        self.N = len(self.locations)
        self.best_replenishment_time = [0]
        self.locations = np.array(self.locations)
        self.distance_timecost_matrix_v = np.zeros((self.N,self.N))
        self.distance_timecost_matrix_h = np.zeros((self.N,self.N))
        for from_index in range(self.N):
            for to_index in range(self.N):
                self.distance_timecost_matrix_v[from_index][to_index] = \
                    calculate_distance(self.locations[from_index], self.locations[to_index])/self.kv
                self.distance_timecost_matrix_h[from_index][to_index] = \
                    calculate_distance(self.locations[from_index], self.locations[to_index]) / self.hv
        self.inventory_cost_matrix_zero = 99999999999 * np.ones(self.N)
        self.shelter_order_size_matrix_zero = np.zeros(self.N)
        self.inventory_cost_matrix_one = 99999999999 * np.ones((self.N, self.T + 1))
        self.shelter_order_size_matrix_one = np.zeros((self.N,self.T+1))
        self.inventory_cost_matrix_two = 99999999999 * np.ones((self.N, self.T + 1, self.T + 1))
        self.shelter_order_size_matrix_two_q_f = np.zeros((self.N,self.T+1,self.T+1))
        self.shelter_order_size_matrix_two_q_s = np.zeros((self.N, self.T + 1, self.T + 1))
        for i in range(1, self.N):
            inventory_cost = 0
            for t in range(1, self.T+1):
                inventory_cost += max(self.h*(self.initial_inventory_level[i]- self.calculate_D_min(t)),
                                          -1*self.s*(self.initial_inventory_level[i] - self.calculate_D_max(t)))
            if inventory_cost == 0:
                self.inventory_cost_matrix_zero[i] = 99999999999
            else:
                self.inventory_cost_matrix_zero[i] = inventory_cost
        for i in range(1, self.N):
            for l in range(1, self.T+1):
                inventory_cost = 0
                q_best = 0
                inventory_cost_best = 99999999999
                for t in range(1, l):
                    inventory_cost += max(self.h*(self.initial_inventory_level[i]-self.calculate_D_min(t)),
                                          -1*self.s*(self.initial_inventory_level[i] - self.calculate_D_max(t)))
                for t_star in range(l, self.T+1):
                    inventory_cost_add = 0
                    q = max(self.calculate_Psi(t_star) - self.initial_inventory_level[i], 0)
                    for t in range(l, self.T+1):
                        inventory_cost_add += max(self.h*(self.initial_inventory_level[i] + q - self.calculate_D_min(t)),
                                                  -1*self.s*(self.initial_inventory_level[i] + q - self.calculate_D_max(t)))
                    if inventory_cost_add <= inventory_cost_best:
                        inventory_cost_best = inventory_cost_add
                        q_best = q
                inventory_cost += inventory_cost_best
                if inventory_cost == 0:
                    self.inventory_cost_matrix_one[i, l] = 99999999999
                else:
                    self.inventory_cost_matrix_one[i,l] = inventory_cost
                    self.shelter_order_size_matrix_one[i,l] = q_best
        for i in range(1, self.N):
            for l_fast in range(1, self.T+1):
                for l_slow in range(l_fast, self.T+1):
                    inventory_cost = 0
                    q_f_best = 0
                    q_s_best = 0
                    inventory_cost_best = 99999999999
                    for t in range(1, l_fast):
                        inventory_cost += max(
                            self.h * (self.initial_inventory_level[i] - self.calculate_D_min(t)),
                            -1 * self.s * (self.initial_inventory_level[i] - self.calculate_D_max(t)))
                    for t_fast in range(l_fast, min(l_slow, self.T)):
                        for t_slow in range(l_slow, self.T + 1):
                            inventory_cost_add = 0
                            q_f = max(self.calculate_Psi(t_fast) - self.initial_inventory_level[i], 0)
                            q_s = max(self.calculate_Psi(t_slow) - q_f - self.initial_inventory_level[i], 0)
                            for t in range(l_fast, l_slow):
                                inventory_cost_add += max(
                                    self.h * (self.initial_inventory_level[i] + q_f - self.calculate_D_min(t)),
                                    -1 * self.s * (self.initial_inventory_level[i] + q_f - self.calculate_D_max(t)))
                            for t in range(l_slow, self.T + 1):
                                inventory_cost_add += max(self.h * (
                                            self.initial_inventory_level[i] + q_f + q_s - self.calculate_D_min(t)),
                                            -1 * self.s * (self.initial_inventory_level[
                                            i] + q_f + q_s - self.calculate_D_max(t)))
                            if inventory_cost_add <= inventory_cost_best:
                                inventory_cost_best = inventory_cost_add
                                q_f_best = q_f
                                q_s_best = q_s
                    inventory_cost += inventory_cost_best
                    if inventory_cost == 0:
                        self.inventory_cost_matrix_two[i, l_fast, l_slow] = 99999999999
                    else:
                        self.inventory_cost_matrix_two[i, l_fast, l_slow] = inventory_cost
                        self.shelter_order_size_matrix_two_q_f[i, l_fast, l_slow] = q_f_best
                        self.shelter_order_size_matrix_two_q_s[i, l_fast, l_slow] = q_s_best

    #--calculate optimal replenishment time
        for i in range(1, self.N):
            index = np.where(self.inventory_cost_matrix_one[i, :] ==
                                                         np.min(self.inventory_cost_matrix_one[i, :]))
            self.best_replenishment_time.append(index[0][-1])

    def calculate_gamma(self, t):
        temp = (self.s - self.h) / (self.s + self.h)
        gamma = math.sqrt(t / (1 - temp * temp))
        gamma = min(gamma, t)
        return gamma

    def calculate_D_min(self, t):
        D_min = t * self.dm - self.calculate_gamma(t) * self.dh
        return D_min

    def calculate_D_max(self, t):
        D_max = t * self.dm + self.calculate_gamma(t) * self.dh
        return D_max

    def calculate_Psi(self, t):
        Psi = (self.s * self.calculate_D_max(t) + self.h * self.calculate_D_min(t)) / (self.h + self.s)
        return Psi


class Solution:
    def __init__(self, problem):
        self.route_v=[]
        self.route_h=[]
        self.removed_pool_v = list(range(1, problem.N))
        self.removed_pool_h = list(range(1, problem.N))
        self.obj = 9999999999999999
        self.shelter_arrival_time_v = np.zeros(problem.N)
        self.shelter_arrival_time_h = np.zeros(problem.N)
        self.shelter_visit_times = np.zeros(problem.N)
        self.shelter_inventory_cost = np.zeros(problem.N)
        self.shelter_order_size_v = np.zeros(problem.N)
        self.shelter_order_size_h = np.zeros(problem.N)



class Removal_operators:
    def __init__(self,removal_operators_list):
        self.removal_operator_score = {}
        self.removal_operator_weight = {}
        self.removal_operator_times = {}
        for i in removal_operators_list:
            self.removal_operator_score[i] = 0
            self.removal_operator_weight[i] = 1
            self.removal_operator_times[i] = 0

    def select(self,removal_operators_list):
        total_weight = sum(self.removal_operator_weight.values())
        probabilities = {}
        for i in removal_operators_list:
            probabilities[i] = self.removal_operator_weight[i]/total_weight
        total_prob = sum(probabilities.values())
        selector = random.random()*total_prob
        cumulative_probability = 0
        for i in removal_operators_list:
            cumulative_probability += probabilities[i]
            if cumulative_probability >= selector:
                selected_operator = i
                break
        return selected_operator

    def update_weight(self, removal_operators_list, problem):
        for i in removal_operators_list:
            if self.removal_operator_times[i] >= 1:
                self.removal_operator_weight[i] = \
                    (1-problem.reaction_factor) * self.removal_operator_weight[i] + \
                    problem.reaction_factor * self.removal_operator_score[i] / self.removal_operator_times[i]
        #reset score and times
        for i in removal_operators_list:
            self.removal_operator_score[i] = 0
            self.removal_operator_times[i] = 0
        return


class Insertion_operators:
    def __init__(self, insertion_operators_list):
        self.insertion_operator_score = {}
        self.insertion_operator_weight = {}
        self.insertion_operator_times = {}
        for i in insertion_operators_list:
            self.insertion_operator_score[i] = 0
            self.insertion_operator_weight[i] = 1
            self.insertion_operator_times[i] = 0

    def select(self,insertion_operators_list):
        total_weight = sum(self.insertion_operator_weight.values())
        probabilities = {}
        for i in insertion_operators_list:
            probabilities[i] = self.insertion_operator_weight[i] / total_weight
        total_prob = sum(probabilities.values())
        selector = random.random() * total_prob
        cumulative_probability = 0
        for i in insertion_operators_list:
            cumulative_probability += probabilities[i]
            if cumulative_probability >= selector:
                selected_operator = i
                break
        return selected_operator

    def update_weight(self,insertion_operators_list, problem):
        for i in insertion_operators_list:
            if self.insertion_operator_times[i] >= 1:
                self.insertion_operator_weight[i] = \
                    (1 - problem.reaction_factor)* self.insertion_operator_weight[i] + \
                    problem.reaction_factor * self.insertion_operator_score[i] / self.insertion_operator_times[i]
        # reset score and times
        for i in insertion_operators_list:
            self.insertion_operator_score[i] = 0
            self.insertion_operator_times[i] = 0
        return


class Record:
    def __init__(self, removal_operators_list, insertion_operators_list):
        self.best_solution_obj_record = []
        self.initial_generator_time_cost = 0
        self.total_time_cost = 0
        self.removal_operator_weights_record = {}
        self.removal_operator_times_record = {}
        self.removal_operator_score_record = {}
        for i in removal_operators_list:
            self.removal_operator_weights_record[i] = []
            self.removal_operator_times_record[i] = []
            self.removal_operator_score_record[i] = []
        self.insertion_operator_weights_record = {}
        self.insertion_operator_times_record = {}
        self.insertion_operator_score_record = {}
        for i in insertion_operators_list:
            self.insertion_operator_weights_record[i] = []
            self.insertion_operator_times_record[i] = []
            self.insertion_operator_score_record[i] = []
        self.best_solution_update_iteration_num = []


# -------------------------- Function --------------------------------------

def calculate_distance(point0, point1):
    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]
    return math.sqrt(dx * dx + dy * dy)


def calculate_relatedness(point0, point1, problem,weight1,weight2):  # -------------- 调参！
    relatedness = weight1*problem.distance_timecost_matrix_v[point0,point1]/(1000/problem.kv) + \
                abs(weight2*(problem.best_replenishment_time[point0]-problem.best_replenishment_time[point1])/problem.T)
    return relatedness


def generate_problem(instance):
    locations = []
    initial_inventoy_level = []
    initial_inventoy_level_out = []
    with open('test_set/' + instance, 'r') as file_to_read:
        lines = file_to_read.readlines()
        flg_locations = False
        flg_demands = False
        for line in lines:
            if "NAME" in line:
                name = re.findall(r"NAME : A-(.+?)-k", line)
            if "NODE_COORD_SECTION" in line:
                flg_locations = True
            if "DEMAND_SECTION" in line:
                flg_locations = False
                flg_demands = True
            if "DEPOT_SECTION" in line:
                flg_demands = False
            if flg_locations:
                temp = line[:-1].split(' ')
                temp.pop(0)
                temp.pop(0)
                locations.append(temp)
            if flg_demands:
                temp = line[:-1].split(' ')
                temp.pop(0)
                temp.pop(-1)
                initial_inventoy_level.append(temp)
        locations.pop(0)
        initial_inventoy_level.pop(0)
        for i in range(len(locations)):
            for j in range(len(locations[i])):
                locations[i][j] = int(locations[i][j])
        for i in range(len(initial_inventoy_level)):
            for j in range(len(initial_inventoy_level[i])):
                initial_inventoy_level_out.append(int(initial_inventoy_level[i][j]))
    problem = Problem(name, locations, initial_inventoy_level_out)
    return problem


def calculate_objective(solution, problem):
    shelter_arrival_time_v = np.zeros(problem.N)
    shelter_arrival_time_h = np.zeros(problem.N)
    shelters_visit_times = np.zeros(problem.N)
    from_v = 0
    from_h = 0
    inventory_cost = 0
    for i in range(len(solution.route_v)):
        shelter_arrival_time_v[solution.route_v[i]] = shelter_arrival_time_v[from_v] + \
                                                       problem.distance_timecost_matrix_v[from_v, solution.route_v[i]]
        shelters_visit_times[solution.route_v[i]] += 1
        from_v = solution.route_v[i]
    for i in range(len(solution.route_h)):
        shelter_arrival_time_h[solution.route_h[i]] = shelter_arrival_time_h[from_h] + \
                                                       problem.distance_timecost_matrix_h[from_h, solution.route_h[i]]
        shelters_visit_times[solution.route_h[i]] += 1
        from_h = solution.route_h[i]
    solution.shelter_visit_times = copy.deepcopy(shelters_visit_times)
    solution.shelter_arrival_time_v = copy.deepcopy(shelter_arrival_time_v)
    solution.shelter_arrival_time_h = copy.deepcopy(shelter_arrival_time_h)
    finish_time_v = shelter_arrival_time_v[from_v] + problem.distance_timecost_matrix_v[from_v, 0]
    finish_time_h = shelter_arrival_time_h[from_h] + problem.distance_timecost_matrix_h[from_h, 0]
    route_cost_v = problem.f_v * finish_time_v
    route_cost_h = problem.f_h * finish_time_h
    for i in range(1, problem.N):
        if shelters_visit_times[i] == 0:
            solution.shelter_inventory_cost[i] = problem.inventory_cost_matrix_zero[i]
            inventory_cost += solution.shelter_inventory_cost[i]
        elif shelters_visit_times[i] == 1:
            solution.shelter_inventory_cost[i] = problem.inventory_cost_matrix_one[i, math.ceil(min(max(shelter_arrival_time_v[i], shelter_arrival_time_h[i]),problem.T))]
            inventory_cost += solution.shelter_inventory_cost[i]
        elif shelters_visit_times[i] == 2:
            solution.shelter_inventory_cost[i] = problem.inventory_cost_matrix_two[i,
                math.ceil(min(shelter_arrival_time_v[i], shelter_arrival_time_h[i], problem.T)),
                math.ceil(min(max(shelter_arrival_time_v[i], shelter_arrival_time_h[i]),problem.T))]
            inventory_cost += solution.shelter_inventory_cost[i]
        else:
            print('Invalid shelter visit times for shelter No. ', i)
    total_cost = route_cost_v+route_cost_h+inventory_cost
    return total_cost

def calculate_remove_cost(solution, index,  mode, problem):
    if mode == 'vehicle':
        from_index = index -1
        to_index = index+1
        if from_index < 0:
            from_node = 0
        else:
            from_node = solution.route_v[from_index]
        if to_index >=len(solution.route_v):
            to_node = 0
        else:
            to_node = solution.route_v[to_index]
        shelter_arrival_time_update_to_node = problem.distance_timecost_matrix_v[from_node,to_node]-problem.distance_timecost_matrix_v[from_node, solution.route_v[index]]- problem.distance_timecost_matrix_v[solution.route_v[index],to_node]
        remove_cost = problem.f_v*shelter_arrival_time_update_to_node#update routing cost
        #update inventory_cost
        for i in range (index,len(solution.route_v)):
            if i == index:
                if solution.shelter_visit_times[solution.route_v[i]] == 2:
                    shelter_inventory_cost_new = problem.inventory_cost_matrix_one[solution.route_v[i], math.ceil(min(solution.shelter_arrival_time_h[solution.route_v[i]],problem.T))]
                    remove_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_v[i]]
                elif solution.shelter_visit_times[solution.route_v[i]] == 1:
                    shelter_inventory_cost_new = problem.inventory_cost_matrix_zero[solution.route_v[i]]
                    remove_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_v[i]]
                else:
                    print('Invalid visit times!')
            else:
                #update arrival time
                shelter_arrival_time_new =  solution.shelter_arrival_time_v[solution.route_v[i]] + shelter_arrival_time_update_to_node
                if math.ceil(min(shelter_arrival_time_new, problem.T)) == math.ceil(min(solution.shelter_arrival_time_v[solution.route_v[i]],problem.T)):
                    remove_cost += 0
                else:
                    if solution.shelter_visit_times[solution.route_v[i]] == 2:
                        shelter_inventory_cost_new = problem.inventory_cost_matrix_two[solution.route_v[i],
                                                                                       math.ceil(min(shelter_arrival_time_new, solution.shelter_arrival_time_h[solution.route_v[i]])),
                                                                                       math.ceil(min(max(shelter_arrival_time_new,solution.shelter_arrival_time_h[solution.route_v[i]]),problem.T))]
                        remove_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_v[i]]
                    elif solution.shelter_visit_times[solution.route_v[i]] == 1:
                        shelter_inventory_cost_new = problem.inventory_cost_matrix_one[solution.route_v[i],
                                                                                       math.ceil(min(shelter_arrival_time_new,problem.T))]
                        remove_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_v[i]]
                    else:
                        print('Invalid visit times!')
    elif mode == 'helicopter':
        from_index = index -1
        to_index = index+1
        if from_index< 0:
            from_node = 0
        else:
            from_node = solution.route_h[from_index]
        if to_index >= len(solution.route_h):
            to_node = 0
        else:
            to_node = solution.route_h[to_index]
        shelter_arrival_time_update_to_node = problem.distance_timecost_matrix_h[from_node,to_node] - problem.distance_timecost_matrix_h[from_node, solution.route_h[index]] - problem.distance_timecost_matrix_h[solution.route_h[index], to_node]
        remove_cost = problem.f_h * shelter_arrival_time_update_to_node
        for i in range(index,len(solution.route_h)):
            if i == index:
                if solution.shelter_visit_times[solution.route_h[i]] == 2:
                    shelter_inventory_cost_new = problem.inventory_cost_matrix_one[solution.route_h[i], math.ceil(min(solution.shelter_arrival_time_v[solution.route_h[i]],problem.T))]
                    remove_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_h[i]]
                elif solution.shelter_visit_times[solution.route_h[i]] == 1:
                    shelter_inventory_cost_new = problem.inventory_cost_matrix_zero[solution.route_h[i]]
                    remove_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_h[i]]
                else:
                    print('Invalid visit times!')
            else:
                shelter_arrival_time_new = solution.shelter_arrival_time_h[solution.route_h[i]] + shelter_arrival_time_update_to_node
                if math.ceil(min(shelter_arrival_time_new,problem.T)) == math.ceil(min(solution.shelter_arrival_time_h[solution.route_h[i]],problem.T)):
                    remove_cost += 0
                else:
                    if solution.shelter_visit_times[solution.route_h[i]] == 2:
                        shelter_inventory_cost_new = problem.inventory_cost_matrix_two[solution.route_h[i],
                                                                                       math.ceil(min(solution.shelter_arrival_time_v[solution.route_h[i]],shelter_arrival_time_new)),
                                                                                       math.ceil(min(max(solution.shelter_arrival_time_v[solution.route_h[i]],shelter_arrival_time_new),problem.T))]
                        remove_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_h[i]]
                    elif solution.shelter_visit_times[solution.route_h[i]] == 1:
                        shelter_inventory_cost_new = problem.inventory_cost_matrix_one[solution.route_h[i],
                                                                                       math.ceil(min(shelter_arrival_time_new,problem.T))]
                        remove_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_h[i]]
                    else:
                        print('Invalid visit times!')
    else:
        print('Invalid mode!')
    return remove_cost

def calculate_insert_cost(solution, index, inserted_shelter, mode, problem):
    if isinstance(inserted_shelter, int):
        if mode == 'vehicle':
            from_index = index -1
            to_index = index
            if from_index < 0:
                from_node = 0
            else:
                from_node = solution.route_v[from_index]
            if to_index >= len(solution.route_v):
                to_node = 0
            else:
                to_node = solution.route_v[to_index]
            inserted_shelter_arrival_time = solution.shelter_arrival_time_v[from_node] + problem.distance_timecost_matrix_v[from_node, inserted_shelter]
            shelter_arrival_time_update_to_node = problem.distance_timecost_matrix_v[from_node,inserted_shelter] + problem.distance_timecost_matrix_v[inserted_shelter,to_node] - problem.distance_timecost_matrix_v[from_node,to_node]
            insert_cost = problem.f_v * shelter_arrival_time_update_to_node
            if solution.shelter_visit_times[inserted_shelter] == 0:
                shelter_inventory_cost_new = problem.inventory_cost_matrix_one[inserted_shelter, math.ceil(min(inserted_shelter_arrival_time,problem.T))]
                insert_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[inserted_shelter]
            elif solution.shelter_visit_times[inserted_shelter] == 1:
                shelter_inventory_cost_new = problem.inventory_cost_matrix_two[inserted_shelter,
                                                                               math.ceil(min(inserted_shelter_arrival_time, solution.shelter_arrival_time_h[inserted_shelter])),
                                                                               math.ceil(min(max(inserted_shelter_arrival_time, solution.shelter_arrival_time_h[inserted_shelter]), problem.T))]
                insert_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[inserted_shelter]
            else:
                print('Invalid visit times!')
            for i in range(index,len(solution.route_v)):
                shelter_arrival_time_new = solution.shelter_arrival_time_v[solution.route_v[i]] +shelter_arrival_time_update_to_node
                if math.ceil(min(shelter_arrival_time_new,problem.T)) == math.ceil(min(solution.shelter_arrival_time_v[solution.route_v[i]], problem.T)):
                    insert_cost += 0
                else:
                    if solution.shelter_visit_times[solution.route_v[i]] == 2:
                        shelter_inventory_cost_new = problem.inventory_cost_matrix_two[solution.route_v[i],
                                                                                       math.ceil(min(shelter_arrival_time_new,solution.shelter_arrival_time_h[solution.route_v[i]])),
                                                                                       math.ceil(min(max(shelter_arrival_time_new,solution.shelter_arrival_time_h[solution.route_v[i]]),problem.T))]
                        insert_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_v[i]]
                    elif solution.shelter_visit_times[solution.route_v[i]] == 1:
                        shelter_inventory_cost_new = problem.inventory_cost_matrix_one[solution.route_v[i],
                                                                                       math.ceil(min(shelter_arrival_time_new,problem.T))]
                        insert_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_v[i]]
                    else:
                        print('Invalid visit times!')
        elif mode == 'helicopter':
            from_index = index -1
            to_index = index
            if from_index < 0:
                from_node = 0
            else:
                from_node = solution.route_h[from_index]
            if to_index >= len(solution.route_h):
                to_node = 0
            else:
                to_node = solution.route_h[to_index]
            inserted_shelter_arrival_time = solution.shelter_arrival_time_h[from_node] + problem.distance_timecost_matrix_h[from_node, inserted_shelter]
            shelter_arrival_time_update_to_node = problem.distance_timecost_matrix_h[from_node,inserted_shelter] + problem.distance_timecost_matrix_h[inserted_shelter, to_node] - problem.distance_timecost_matrix_h[from_node, to_node]
            insert_cost = problem.f_h * shelter_arrival_time_update_to_node
            if solution.shelter_visit_times[inserted_shelter] == 0:
                shelter_inventory_cost_new = problem.inventory_cost_matrix_one[inserted_shelter, math.ceil(min(inserted_shelter_arrival_time,problem.T))]
                insert_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[inserted_shelter]
            elif solution.shelter_visit_times[inserted_shelter] == 1:
                shelter_inventory_cost_new = problem.inventory_cost_matrix_two[inserted_shelter,
                                                                               math.ceil(min(solution.shelter_arrival_time_v[inserted_shelter], inserted_shelter_arrival_time)),
                                                                               math.ceil(min(max(solution.shelter_arrival_time_v[inserted_shelter], inserted_shelter_arrival_time),problem.T))]
                insert_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[inserted_shelter]
            else:
                print('Invalid visit times!')
            for i in range(index, len(solution.route_h)):
                shelter_arrival_time_new = solution.shelter_arrival_time_h[solution.route_h[i]] + shelter_arrival_time_update_to_node
                if math.ceil(min(shelter_arrival_time_new, problem.T)) == math.ceil(min(solution.shelter_arrival_time_h[solution.route_h[i]],problem.T)):
                    insert_cost += 0
                else:
                    if solution.shelter_visit_times[solution.route_h[i]] == 2:
                        shelter_inventory_cost_new = problem.inventory_cost_matrix_two[solution.route_h[i],
                                                                                       math.ceil(min(solution.shelter_arrival_time_v[solution.route_h[i]],shelter_arrival_time_new)),
                                                                                       math.ceil(min(max(solution.shelter_arrival_time_v[solution.route_h[i]], shelter_arrival_time_new), problem.T))]
                        insert_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_h[i]]
                    elif solution.shelter_visit_times[solution.route_h[i]] ==1:
                        shelter_inventory_cost_new = problem.inventory_cost_matrix_one[solution.route_h[i],
                                                                                       math.ceil(min(shelter_arrival_time_new,problem.T))]
                        insert_cost += shelter_inventory_cost_new - solution.shelter_inventory_cost[solution.route_h[i]]
                    else:
                        print('Invalid visit times!')
        else:
            print('Invalid mode!')
    else:
        print('Invalid insert shelter index!')
    return insert_cost


def generate_initial_solution(problem):
    solution = Solution(problem)
    insertion_cost = -1
    solution.obj = calculate_objective(solution,problem)
    while insertion_cost <= 0:
        best_cost_gap = 999999
        for i in range(len(solution.removed_pool_v)):
            cost_gap = calculate_insert_cost(solution,len(solution.route_v), solution.removed_pool_v[i], 'vehicle', problem)
            if cost_gap <= best_cost_gap:
                best_cost_gap = cost_gap
                best_i = i
        insertion_cost_v = best_cost_gap
        if insertion_cost_v <= 0:
            solution.route_v.append(solution.removed_pool_v[best_i])
            solution.removed_pool_v.pop(best_i)
            solution.obj = calculate_objective(solution, problem)
        best_cost_gap = 999999
        for i in range(len(solution.removed_pool_h)):
            cost_gap = calculate_insert_cost(solution,len(solution.route_h), solution.removed_pool_h[i], 'helicopter',problem)
            if cost_gap <= best_cost_gap:
                best_cost_gap = cost_gap
                best_i = i
        insertion_cost_h = best_cost_gap
        if insertion_cost_h <= 0:
            solution.route_h.append(solution.removed_pool_h[best_i])
            solution.removed_pool_h.pop(best_i)
            solution.obj = calculate_objective(solution, problem)
        insertion_cost = min(insertion_cost_v,insertion_cost_h)
    return solution


def do_remove(operator_name, solution, problem):
    remove_num_v = math.ceil(len(solution.route_v) * problem.removal_rate)
    remove_num_h = math.ceil(len(solution.route_h) * problem.removal_rate)
    remove_num = remove_num_v+remove_num_h
    removed_num_v = 0
    removed_num_h = 0
    removed_num = 0
    if operator_name == 'shaw':
        while removed_num_v <= remove_num_v:
            if not solution.route_v:
                break
            if not solution.removed_pool_v:
                remove_index = random.randint(0,len(solution.route_v)-1)
                remove = solution.route_v[remove_index]
                solution.route_v.pop(remove_index)
                solution.removed_pool_v.append(remove)
                removed_num_v += 1
            else:
                relatedness_min=999999
                for i in range(len(solution.removed_pool_v)):
                    for j in range(len(solution.route_v)):
                        relatedness = calculate_relatedness(solution.removed_pool_v[i], solution.route_v[j], problem, 0.5, 0.5)
                        if relatedness <= relatedness_min:
                            relatedness_min = relatedness
                            remove_index = j
                remove = solution.route_v[remove_index]
                solution.route_v.pop(remove_index)
                solution.removed_pool_v.append(remove)
                removed_num_v += 1
                if remove in solution.route_h:
                    remove_index = solution.route_h.index(remove)
                    solution.route_h.pop(remove_index)
                    solution.removed_pool_h.append(remove)
                    removed_num_h += 1
        while removed_num_h <= remove_num_h:
            if not solution.route_h:
                break
            if not solution.removed_pool_h:
                remove_index = random.randint(0, len(solution.route_h)-1)
                remove = solution.route_h[remove_index]
                solution.route_h.pop(remove_index)
                solution.removed_pool_h.append(remove)
                removed_num_h += 1
            else:
                relatedness_min = 999999
                for i in range(len(solution.removed_pool_h)):
                    for j in range(len(solution.route_h)):
                        relatedness = calculate_relatedness(solution.removed_pool_h[i], solution.route_h[j], problem, 0.5, 0.5)
                        if relatedness <= relatedness_min:
                            relatedness_min = relatedness
                            remove_index = j
                remove = solution.route_h[remove_index]
                solution.route_h.pop(remove_index)
                solution.removed_pool_h.append(remove)
                removed_num_h += 1
                if remove in solution.route_v:
                    remove_index = solution.route_v.index(remove)
                    solution.route_v.pop(remove_index)
                    solution.removed_pool_v.append(remove)
                    removed_num_v += 1
    elif operator_name == 'distance':
        while removed_num_v <= remove_num_v:
            if not solution.route_v:
                break
            if not solution.removed_pool_v:
                remove_index = random.randint(0,len(solution.route_v)-1)
                remove = solution.route_v[remove_index]
                solution.route_v.pop(remove_index)
                solution.removed_pool_v.append(remove)
                removed_num_v += 1
            else:
                relatedness_min=999999
                for i in range(len(solution.removed_pool_v)):
                    for j in range(len(solution.route_v)):
                        relatedness = calculate_relatedness(solution.removed_pool_v[i], solution.route_v[j], problem, 1, 0)
                        if relatedness <= relatedness_min:
                            relatedness_min = relatedness
                            remove_index = j
                remove = solution.route_v[remove_index]
                solution.route_v.pop(remove_index)
                solution.removed_pool_v.append(remove)
                removed_num_v += 1
        while removed_num_h <= remove_num_h:
            if not solution.route_h:
                break
            if not solution.removed_pool_h:
                remove_index = random.randint(0, len(solution.route_h)-1)
                remove = solution.route_h[remove_index]
                solution.route_h.pop(remove_index)
                solution.removed_pool_h.append(remove)
                removed_num_h += 1
            else:
                relatedness_min = 999999
                for i in range(len(solution.removed_pool_h)):
                    for j in range(len(solution.route_h)):
                        relatedness = calculate_relatedness(solution.removed_pool_h[i], solution.route_h[j], problem, 1, 0)
                        if relatedness <= relatedness_min:
                            relatedness_min = relatedness
                            remove_index = j
                remove = solution.route_h[remove_index]
                solution.route_h.pop(remove_index)
                solution.removed_pool_h.append(remove)
                removed_num_h += 1
    elif operator_name == 'worst':
        solution.obj = calculate_objective(solution, problem)
        while removed_num <= remove_num:
            if not solution.route_v and not solution.route_h:
                break
            best_cost_gap = 99999999

            for i in range(len(solution.route_v)):
                if not solution.route_v:
                    break
                cost_gap = calculate_remove_cost(solution, i, 'vehicle', problem)
                if cost_gap < best_cost_gap:
                    best_cost_gap = cost_gap
                    remove_route = 'vehicle'
                    remove_index = i
            for i in range(len(solution.route_h)):
                if not solution.route_h:
                    break
                cost_gap = calculate_remove_cost(solution, i, 'helicopter', problem)
                if cost_gap < best_cost_gap:
                    best_cost_gap = cost_gap
                    remove_route = 'helicopter'
                    remove_index = i
            if remove_route == 'vehicle':
                remove = solution.route_v[remove_index]
                solution.route_v.pop(remove_index)
                solution.removed_pool_v.append(remove)
                removed_num += 1
                solution.obj = calculate_objective(solution,problem)
            elif remove_route == 'helicopter':
                remove = solution.route_h[remove_index]
                solution.route_h.pop(remove_index)
                solution.removed_pool_h.append(remove)
                removed_num += 1
                solution.obj = calculate_objective(solution,problem)
            else:
                print('error: Invalid remove_route Type!')
    elif operator_name == 'worst_replenishment_time':
        while removed_num <= remove_num:
            set_route_v = set(solution.route_v)
            set_route_h = set(solution.route_h)
            set_v = set_route_v.difference(set_route_h)
            set_h = set_route_h.difference(set_route_v)
            if not set_v | set_h:
                break
            arrival_time_v = 0
            arrival_time_h = 0
            worst_arrival_time_gap = 0
            shelter_from = 0
            for i in range(len(solution.route_v)):
                if solution.route_v[i] in set_v:
                    arrival_time_v += problem.distance_timecost_matrix_v[shelter_from,solution.route_v[i]]
                    arrival_time_gap = abs(arrival_time_v-problem.best_replenishment_time[solution.route_v[i]])
                    shelter_from = solution.route_v[i]
                    if arrival_time_gap >= worst_arrival_time_gap:
                        worst_arrival_time_gap = arrival_time_gap
                        remove_index = i
                        remove_route = 'vehicle'
            shelter_from = 0
            for i in range(len(solution.route_h)):
                if solution.route_h[i] in set_h:
                    arrival_time_h += problem.distance_timecost_matrix_h[shelter_from,solution.route_h[i]]
                    arrival_time_gap = abs(arrival_time_h - problem.best_replenishment_time[solution.route_h[i]])
                    shelter_from = solution.route_h[i]
                    if arrival_time_gap >= worst_arrival_time_gap:
                        worst_arrival_time_gap = arrival_time_gap
                        remove_index = i
                        remove_route = 'helicopter'
            if remove_route == 'vehicle':
                remove = solution.route_v[remove_index]
                solution.route_v.pop(remove_index)
                solution.removed_pool_v.append(remove)
                removed_num += 1
            elif remove_route == 'helicopter':
                remove = solution.route_h[remove_index]
                solution.route_h.pop(remove_index)
                solution.removed_pool_h.append(remove)
                removed_num += 1
            else:
                print('error: Invalid remove_route Type!')
    elif operator_name == 'random':
        while removed_num <= remove_num:
            if not solution.route_v and not solution.route_h:
                break
            total_index = random.randint(0, len(solution.route_v) + len(solution.route_h)-1)
            if total_index <= len(solution.route_v)-1:
                remove_index = total_index
                remove_route = 'vehicle'
            else:
                remove_index = total_index - len(solution.route_v)
                remove_route = 'helicopter'
            if remove_route == 'vehicle':
                remove = solution.route_v[remove_index]
                solution.route_v.pop(remove_index)
                solution.removed_pool_v.append(remove)
                removed_num += 1
            elif remove_route == 'helicopter':
                remove = solution.route_h[remove_index]
                solution.route_h.pop(remove_index)
                solution.removed_pool_h.append(remove)
                removed_num += 1
    else:
        print('error: Invalid Removal Operator Name')
    return


def do_insert(operator_name, solution, problem):
    if operator_name == 'greedy':
        insertion_cost = -1
        solution.obj = calculate_objective(solution, problem)
        while insertion_cost < 0:
            best_cost_gap = 999999
            for i in range(len(solution.removed_pool_v)):
                for j in range(len(solution.route_v)+1):
                    cost_gap = calculate_insert_cost(solution,j,solution.removed_pool_v[i],'vehicle',problem)
                    if cost_gap < best_cost_gap:
                        best_cost_gap = cost_gap
                        best_i = i
                        best_j = j
                        insert_route = 'vehicle'
            for i in range(len(solution.removed_pool_h)):
                for j in range(len(solution.route_h)+1):
                    cost_gap = calculate_insert_cost(solution,j,solution.removed_pool_h[i],'helicopter',problem)
                    if cost_gap < best_cost_gap:
                        best_cost_gap = cost_gap
                        best_i = i
                        best_j = j
                        insert_route = 'helicopter'
            insertion_cost = best_cost_gap
            if insertion_cost < 0:
                if insert_route == 'vehicle':
                    solution.route_v.insert(best_j, solution.removed_pool_v[best_i])
                    solution.removed_pool_v.pop(best_i)
                    solution.obj = calculate_objective(solution, problem)
                elif insert_route == 'helicopter':
                    solution.route_h.insert(best_j, solution.removed_pool_h[best_i])
                    solution.removed_pool_h.pop(best_i)
                    solution.obj = calculate_objective(solution, problem)
                else:
                    print('error: Invalid Insert_route Type!')
    elif operator_name == '2_regret':
        insertion_cost = -1
        solution.obj = calculate_objective(solution, problem)
        while insertion_cost < 0:
            best_regret = 0
            insertion_cost = 999999
            for i in range(len(solution.removed_pool_v)):
                if not not solution.route_v:
                    best_cost_gap = 999999
                    second_cost_gap = 999999
                    for j in range(len(solution.route_v)+1):
                        cost_gap = calculate_insert_cost(solution, j, solution.removed_pool_v[i],'vehicle',problem)
                        if cost_gap < second_cost_gap:
                            if cost_gap < best_cost_gap:
                                best_cost_gap = cost_gap
                                best_i = i
                                best_j = j
                            else:
                                second_cost_gap = cost_gap
                    regret = second_cost_gap - best_cost_gap
                    if regret > best_regret:
                        best_regret = regret
                        insert_route = 'vehicle'
                        insertion_cost = best_cost_gap
                        insert_i = best_i
                        insert_j = best_j
            for i in range(len(solution.removed_pool_h)):
                if not not solution.route_h:
                    best_cost_gap = 999999
                    second_cost_gap = 999999
                    for j in range(len(solution.route_h)+1):
                        cost_gap = calculate_insert_cost(solution,j,solution.removed_pool_h[i],'helicopter',problem)
                        if cost_gap < second_cost_gap:
                            if cost_gap < best_cost_gap:
                                best_cost_gap = cost_gap
                                best_i = i
                                best_j = j
                            else:
                                second_cost_gap = cost_gap
                    regret = second_cost_gap - best_cost_gap
                    if regret > best_regret:
                        best_regret = regret
                        insert_route = 'helicopter'
                        insertion_cost = best_cost_gap
                        insert_i = best_i
                        insert_j = best_j
            if insertion_cost < 0:
                if insert_route == 'vehicle':
                    solution.route_v.insert(insert_j,solution.removed_pool_v[insert_i])
                    solution.removed_pool_v.pop(insert_i)
                    solution.obj = calculate_objective(solution, problem)
                elif insert_route == 'helicopter':
                    solution.route_h.insert(insert_j,solution.removed_pool_h[insert_i])
                    solution.removed_pool_h.pop(insert_i)
                    solution.obj = calculate_objective(solution, problem)
                else:
                    print('error: Invalid Insert_route Type!')
    elif operator_name == 'shaw':
        set_v = set(solution.removed_pool_v)
        set_h = set(solution.removed_pool_h)
        candidate_pool = list(set_v & set_h)
        insertion_cost = -1
        while insertion_cost < 0:
            if len(candidate_pool) <= 1:
                break
            cost_current = calculate_objective(solution, problem)
            best_cost_gap = 999999
            selected_index = random.randint(0, len(candidate_pool) - 1)
            selected_shelter_A = candidate_pool[selected_index]
            candidate_pool.pop(selected_index)
            best_relatedness = 999999
            for i in range(len(candidate_pool)):
                relatedness = calculate_relatedness(selected_shelter_A, candidate_pool[i], problem, 0.5, 0.5)
                if relatedness <= best_relatedness:
                    best_relatedness = relatedness
                    selected_shelter_B = candidate_pool[i]
                    candidate_pool_pop_index = i
            candidate_pool.pop(candidate_pool_pop_index)
            selected_shelter = [selected_shelter_A, selected_shelter_B]
            for i in range(len(solution.route_v)+1):
                solution.route_v.insert(i, selected_shelter[1])
                solution.route_v.insert(i, selected_shelter[0])
                cost_gap = calculate_objective(solution, problem) - cost_current
                solution.route_v.pop(i)
                solution.route_v.pop(i)
                if cost_gap < best_cost_gap:
                    best_cost_gap = cost_gap
                    best_i = i
                    best_selected_shelter = copy.deepcopy(selected_shelter)
                    insert_route = 'vehicle'
            selected_shelter = [selected_shelter_B, selected_shelter_A]
            for i in range(len(solution.route_v)+1):
                solution.route_v.insert(i, selected_shelter[1])
                solution.route_v.insert(i, selected_shelter[0])
                cost_gap = calculate_objective(solution, problem) - cost_current
                solution.route_v.pop(i)
                solution.route_v.pop(i)
                if cost_gap < best_cost_gap:
                    best_cost_gap = cost_gap
                    best_i = i
                    best_selected_shelter = copy.deepcopy(selected_shelter)
                    insert_route = 'vehicle'
            selected_shelter = [selected_shelter_A, selected_shelter_B]
            for i in range(len(solution.route_h)+1):
                solution.route_h.insert(i, selected_shelter[1])
                solution.route_h.insert(i, selected_shelter[0])
                cost_gap = calculate_objective(solution, problem) - cost_current
                solution.route_h.pop(i)
                solution.route_h.pop(i)
                if cost_gap < best_cost_gap:
                    best_cost_gap = cost_gap
                    best_i = i
                    best_selected_shelter = copy.deepcopy(selected_shelter)
                    insert_route = 'helicopter'
            selected_shelter = [selected_shelter_B, selected_shelter_A]
            for i in range(len(solution.route_h)+1):
                solution.route_h.insert(i, selected_shelter[1])
                solution.route_h.insert(i, selected_shelter[0])
                cost_gap = calculate_objective(solution, problem) - cost_current
                solution.route_h.pop(i)
                solution.route_h.pop(i)
                if cost_gap < best_cost_gap:
                    best_cost_gap = cost_gap
                    best_i = i
                    best_selected_shelter = copy.deepcopy(selected_shelter)
                    insert_route = 'helicopter'
            insertion_cost = best_cost_gap
            if insertion_cost < 0:
                pop_index = []
                if insert_route == 'vehicle':
                    solution.route_v.insert(best_i, best_selected_shelter[1])
                    solution.route_v.insert(best_i, best_selected_shelter[0])
                    for i in range(len(solution.removed_pool_v)):
                        if solution.removed_pool_v[i] in best_selected_shelter:
                            pop_index.append(i)
                    solution.removed_pool_v.pop(pop_index[1])
                    solution.removed_pool_v.pop(pop_index[0])
                elif insert_route == 'helicopter':
                    solution.route_h.insert(best_i, best_selected_shelter[1])
                    solution.route_h.insert(best_i, best_selected_shelter[0])
                    for i in range(len(solution.removed_pool_h)):
                        if solution.removed_pool_h[i] in best_selected_shelter:
                            pop_index.append(i)
                    solution.removed_pool_h.pop(pop_index[1])
                    solution.removed_pool_h.pop(pop_index[0])
                else:
                    print('error: Invalid Insert_route Type!')
        solution.obj = calculate_objective(solution,problem)
    else:
        print('error: Invalid insertion Operator Name')
    return


def do_2opt(best_solution, current_solution, problem):
    unupdate_num = 0
    while unupdate_num <= 200:
        if random.randint(0, 1) == 0:
            if len(best_solution.route_v) >= 2 :
                route = copy.deepcopy(best_solution.route_v)
                solution_new = copy.deepcopy(best_solution)
                index1 = random.randint(0, len(route))
                index2 = random.randint(0, len(route))
                min_index = min(index1, index2)
                max_index = max(index1, index2)
                part1 = route[0:min_index]
                part2 = route[min_index: max_index]
                part3 = route[max_index: len(best_solution.route_v)]
                route_new = part1 + part2[::-1] + part3
                solution_new.route_v = route_new
            else:
                solution_new = copy.deepcopy(best_solution)
        else:
            if len(best_solution.route_h) >= 2:
                route = copy.deepcopy(best_solution.route_h)
                solution_new = copy.deepcopy(best_solution)
                index1 = random.randint(0, len(route))
                index2 = random.randint(0, len(route))
                min_index = min(index1, index2)
                max_index = max(index1, index2)
                part1 = route[0:min_index]
                part2 = route[min_index: max_index]
                part3 = route[max_index: len(best_solution.route_v)]
                route_new = part1 + part2[::-1] + part3
                solution_new.route_h = route_new
            else:
                solution_new = copy.deepcopy(best_solution)
        solution_new.obj = calculate_objective(solution_new, problem)

        if solution_new.obj < best_solution.obj:
            best_solution = copy.deepcopy(solution_new)
            current_solution = copy.deepcopy(current_solution)
            print('best_solution updated! Best_soltuion = ', best_solution.obj)
            unupdate_num = 0
        else:
            unupdate_num += 1
    return best_solution, current_solution

