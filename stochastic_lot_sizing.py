# Reference: 512-F20-Topic7 Stochastic Lot-Sizing without Backlog
import math
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict


# Problem Data Input
class Param:
    setup_cost: float
    production_cost: float
    inventory_cost: float
    unit_revenue: float
    salvage_price: float
    inventory_limit: int
    production_limit: int
    time_horizon: int
    penalty: int
    shift_period: int


# Regard 1000 as 1
class ParamInstance(Param):
    setup_cost = 2000000
    production_cost = 172800
    inventory_cost = 2100
    unit_revenue = 240000
    salvage_price = 99000
    inventory_limit = 100
    production_limit = 80
    time_horizon = 6
    penalty = 100000
    shift_period = 2


class Demand:
    mandatory: int
    optional_higher_risk: int
    optional_lower_risk: int
    factor_higher_risk: int
    factor_lower_risk: int


class BeforeDemandInstance(Demand):
    mandatory = 24112
    optional_higher_risk = 0
    optional_lower_risk = 30131
    factor_higher_risk = 28
    factor_lower_risk = 5


class AfterDemandInstance(Demand):
    mandatory = 33665
    optional_higher_risk = 19973
    optional_lower_risk = 22367
    factor_higher_risk = 28
    factor_lower_risk = 5


def prob_distribution(x: float) -> float:
    x *= 100
    a, mu, sigma = 3.24, 2.09, 1.04
    y = a * math.exp(-math.pow((x - mu), 2) / (2 * math.pow(sigma, 2))) / 100
    return y


def calculate_demand_distribution(demand: Demand, division: int) -> Dict[int, float]:
    infect_rate = np.linspace(0, 0.1, division)
    prob = []
    total_demand = []
    for rate in infect_rate:
        prob.append(prob_distribution(rate))
        mandatory_demand = demand.mandatory * (1 + rate * 7)
        optional_demand_higher = demand.optional_higher_risk * rate * (demand.factor_higher_risk + 7)
        optional_demand_lower = demand.optional_lower_risk * rate * (demand.factor_lower_risk + 7)
        total_demand.append(int(round(mandatory_demand + optional_demand_higher + optional_demand_lower, -3) / 1000))
    modified_prob = list(map(lambda p: p / sum(prob), prob))

    demand_dist = {}
    for index in range(division):
        if total_demand[index] in demand_dist:
            demand_dist[total_demand[index]] += modified_prob[index]
        else:
            demand_dist[total_demand[index]] = modified_prob[index]

    return demand_dist


# Markov Decision Process
def calculate_lost_demand(inv_limit: int, demand_dist: Dict[int, float]) -> Dict[int, float]:
    lost_demand = {}
    for level in range(inv_limit + 1):
        expected_lost = 0
        for demand, prob in demand_dist.items():
            expected_lost += max(demand - level, 0) * prob
        lost_demand[level] = expected_lost
    return lost_demand


def calculate_revenue(inv_limit: int, unit_revenue: float, demand_dist: Dict[int, float]) -> Dict[int, float]:
    revenue = {}
    for level in range(inv_limit + 1):
        expected_revenue = 0
        for demand, prob in demand_dist.items():
            expected_revenue += min(level, demand) * unit_revenue * prob
        revenue[level] = expected_revenue
    return revenue


def calculate_reward(param: Param, demand_dist: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    lost_demand = calculate_lost_demand(param.inventory_limit, demand_dist)
    revenue = calculate_revenue(param.inventory_limit, param.unit_revenue, demand_dist)
    reward = {}
    for inv_level in range(param.inventory_limit + 1):
        for prod_level in range(min(param.inventory_limit - inv_level, param.production_limit) + 1):
            expected_penalty = lost_demand[inv_level + prod_level] * param.penalty
            expected_revenue = revenue[inv_level + prod_level]
            fixed_cost = 0 if prod_level == 0 else param.setup_cost
            prod_cost = param.production_cost * prod_level
            inv_cost = param.inventory_cost * (inv_level + prod_level)
            reward[(inv_level, prod_level)] = expected_revenue - expected_penalty - fixed_cost - prod_cost - inv_cost
    return reward


def calculate_transition_matrix(inv_limit: int, demand_dist: Dict[int, float]) -> List[List[int]]:
    transition_matrix = [[0 for _ in range(inv_limit + 1)] for _ in range(inv_limit + 1)]
    for level in range(inv_limit + 1):
        for demand, prob in demand_dist.items():
            transition_matrix[level][max(0, level - demand)] += prob
    return transition_matrix


def markov_decision_process(param: Param, before_demand_dist: Dict[int, float],
                            after_demand_dist: Dict[int, float]) -> Tuple[List[List[float]], List[List[int]]]:
    value_res = [[0.0 for _ in range(param.inventory_limit + 1)] for _ in range(param.time_horizon + 1)]
    action_res = [[0 for _ in range(param.inventory_limit + 1)] for _ in range(param.time_horizon + 1)]

    for inv_level in range(param.inventory_limit + 1):
        value_res[param.time_horizon][inv_level] = inv_level * param.salvage_price

    for period in range(param.time_horizon - 1, -1, -1):
        if period < param.shift_period:
            reward = calculate_reward(param, before_demand_dist)
            transition_matrix = calculate_transition_matrix(param.inventory_limit, before_demand_dist)
        else:
            reward = calculate_reward(param, after_demand_dist)
            transition_matrix = calculate_transition_matrix(param.inventory_limit, after_demand_dist)

        for inv_level in range(param.inventory_limit + 1):
            value_compare = {}
            for prod_level in range(min(param.inventory_limit - inv_level, param.production_limit) + 1):
                value = reward[(inv_level, prod_level)]
                for next_level in range(param.inventory_limit + 1):
                    value += value_res[period + 1][next_level] * transition_matrix[inv_level + prod_level][next_level]
                value_compare[prod_level] = value
            action = max(value_compare, key=value_compare.get)
            action_res[period][inv_level] = action
            value_res[period][inv_level] = value_compare[action]

    return value_res, action_res


def calculate_max_lost_demand(param: Param, before_demand_dist: Dict[int, float], after_demand_dist: Dict[int, float],
                              action: List[List[int]]) -> float:
    max_lost = 0
    for period in range(param.time_horizon):
        if period < param.shift_period:
            lost_demand = calculate_lost_demand(param.inventory_limit, before_demand_dist)
        else:
            lost_demand = calculate_lost_demand(param.inventory_limit, after_demand_dist)

        for inv_level in range(param.inventory_limit + 1):
            level = inv_level + action[period][inv_level]
            max_lost = max(max_lost, lost_demand[level])

    return max_lost


if __name__ == '__main__':
    param_instance = ParamInstance()
    before_demand_instance = BeforeDemandInstance()
    before_demand_distribution = calculate_demand_distribution(before_demand_instance, 1000)
    after_demand_instance = AfterDemandInstance()
    after_demand_distribution = calculate_demand_distribution(after_demand_instance, 1000)
    optimal_value, optimal_action = markov_decision_process(param_instance, before_demand_distribution,
                                                            after_demand_distribution)

    max_lost_demand = calculate_max_lost_demand(param_instance, before_demand_distribution, after_demand_distribution,
                                                optimal_action)
    print('maximum lost demand in one epoch:', max_lost_demand)

    optimal_value_matrix = np.transpose(optimal_value)
    optimal_action_matrix = np.transpose(optimal_action)
    print('optimal total net profit in all epochs with no inventory at the beginning:', optimal_value_matrix[0][0])

    data_index = []
    for row in range(param_instance.inventory_limit + 1):
        row_name = 'Inventory Level = ' + str(row)
        data_index.append(row_name)

    data_columns = []
    for column in range(1, param_instance.time_horizon + 1):
        column_name = 'Decision Epoch = ' + str(column)
        data_columns.append(column_name)
    data_columns.append('Dummy Period')

    optimal_value_matrix = pd.DataFrame(data=optimal_value_matrix, index=data_index, columns=data_columns)
    optimal_action_matrix = pd.DataFrame(data=optimal_action_matrix, index=data_index, columns=data_columns)

    optimal_value_matrix.to_csv('optimal_value.csv')
    optimal_action_matrix.to_csv('optimal_action.csv')
