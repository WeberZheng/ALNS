########################################################################################################################
# Robust Optimization for the Dual Sourcing Inventory Routing Problem in Disaster Relief
# Adaptive Large Neighborhood Search Algorithm
# Python>=3.8
# Author: Weibo Zheng
########################################################################################################################

import base
import numpy as np
import math
import re
import csv
import random
import copy
import time
import os
# --------------------------- Global Configuration ---------------------------------------

removal_operators_list=['shaw','distance','worst','worst_replenishment_time','random']
insertion_operators_list=['greedy','2_regret','shaw']

##################################################################################

for root, dirs, files in os.walk('./Test_set'):
    for set_name in files:
        print(set_name)
        time_start = time.time()
        instance = set_name
        problem = base.generate_problem(instance)
        records = base.Record(removal_operators_list, insertion_operators_list)
        solution = base.generate_initial_solution(problem)
        removal_operators = base.Removal_operators(removal_operators_list)
        insertion_operators = base.Insertion_operators(insertion_operators_list)
        solution.obj = base.calculate_objective(solution, problem)
        current_solution = copy.deepcopy(solution)
        best_solution = copy.deepcopy(solution)
        time_end = time.time()
        time_cost = time_end - time_start
        records.initial_generator_time_cost = time_cost
        time_start = time.time()
        iteration_num = 0
        iteration_counter = 0
        segment_counter = 0
        while time_cost <= 3600:
            if problem.temperature <= 0.01:
                break
            if iteration_counter >= problem.segments_capacity:
                segment_counter += 1
                iteration_counter = 0
                for i in removal_operators_list:
                    records.removal_operator_score_record[i].append(removal_operators.removal_operator_score[i])
                    records.removal_operator_weights_record[i].append(removal_operators.removal_operator_weight[i])
                    records.removal_operator_times_record[i].append(removal_operators.removal_operator_times[i])
                for i in insertion_operators_list:
                    records.insertion_operator_score_record[i].append(insertion_operators.insertion_operator_score[i])
                    records.insertion_operator_weights_record[i].append(insertion_operators.insertion_operator_weight[i])
                    records.insertion_operator_times_record[i].append(insertion_operators.insertion_operator_times[i])
                removal_operators.update_weight(removal_operators_list,problem)
                insertion_operators.update_weight(insertion_operators_list,problem)
                for i in removal_operators_list:
                    removal_operators.removal_operator_score[i] = 0
                    removal_operators.removal_operator_times[i] = 0
                for i in insertion_operators_list:
                    insertion_operators.insertion_operator_score[i] = 0
                    insertion_operators.insertion_operator_times[i] = 0
                best_solution, current_solution = base.do_2opt(best_solution, current_solution, problem)
            solution = copy.deepcopy(current_solution)
            selected_removal_operator = removal_operators.select(removal_operators_list)
            selected_insertion_operator = insertion_operators.select(insertion_operators_list)
            removal_operators.removal_operator_times[selected_removal_operator] +=1
            insertion_operators.insertion_operator_times[selected_insertion_operator] +=1
            sub_time_start = time.time()
            sub_time_cost = sub_time_start - time_end
            base.do_remove(selected_removal_operator, solution, problem)
            sub_time_end = time.time()
            sub_time_cost = sub_time_end - sub_time_start
            sub_time_start = sub_time_end
            base.do_insert(selected_insertion_operator, solution, problem)
            sub_time_end = time.time()
            sub_time_cost = sub_time_end - sub_time_start
            sub_time_start = sub_time_end
            records.best_solution_obj_record.append(best_solution.obj)
            if solution.obj < current_solution.obj:
                if solution.obj < best_solution.obj:
                    removal_operators.removal_operator_score[selected_removal_operator] += problem.score_factor_1
                    insertion_operators.insertion_operator_score[selected_insertion_operator] += problem.score_factor_1
                    current_solution = copy.deepcopy(solution)
                    best_solution = copy.deepcopy(solution)
                    current_solution = copy.deepcopy(solution)
                    records.best_solution_obj_record[iteration_num] = best_solution.obj
                    segment_counter = 0
                    print('get best solution= ', best_solution.obj)
                    records.best_solution_update_iteration_num.append(iteration_num)
                else:
                    removal_operators.removal_operator_score[selected_removal_operator] += problem.score_factor_2
                    insertion_operators.insertion_operator_score[selected_insertion_operator] += problem.score_factor_2
                    current_solution = copy.deepcopy(solution)
            elif random.random() < math.exp(-(solution.obj - current_solution.obj)/problem.temperature):
                removal_operators.removal_operator_score[selected_removal_operator] += problem.score_factor_3
                insertion_operators.insertion_operator_score[selected_insertion_operator] += problem.score_factor_3
                current_solution = copy.deepcopy(solution)
            iteration_num += 1
            iteration_counter += 1
            time_end = time.time()
            time_cost = time_end - time_start
            if time_cost >= 1200 and iteration_num <= 3000/3:
                problem.cooling_rate = (0.01/problem.temperature) ** (1/iteration_num)
            if time_cost >= 2400 and iteration_num <= 3000/2:
                problem.cooling_rate = (0.01/problem.temperature) ** (1/(iteration_num/2))
            problem.temperature = problem.temperature*problem.cooling_rate
            print('iteration '+ str(iteration_num)+' finished!')
        print(problem.instance_name,' has finished!')
        with open('Data.csv', 'a', newline='', encoding="ANSI") as f_out:
            csv.writer(f_out, delimiter=',').writerow([set_name, best_solution.obj,time_cost])
        print(problem.instance_name, ' has finished!')
