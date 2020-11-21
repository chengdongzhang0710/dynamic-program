# Reference: 512-F20-Topic7 Stochastic Lot-Sizing without Backlog

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


class Instance(Param):
    setup_cost = 4
    production_cost = 2
    inventory_cost = 1
    unit_revenue = 8
    salvage_price = 0
    inventory_limit = 3
    production_limit = 3
    time_horizon = 3


demand_distribution = {
    0: 0.25,
    1: 0.5,
    2: 0.25,
}


# Markov Decision Process
def calculate_revenue(inv_limit: int, unit_revenue: float, demand_dist: dict) -> dict:
    revenue = {}
    for level in range(inv_limit + 1):
        expected_revenue = 0
        for demand, prob in demand_dist.items():
            expected_revenue += min(level, demand) * unit_revenue * prob
        revenue[level] = expected_revenue
    return revenue


def calculate_reward(param: Param, demand_dist: dict) -> dict:
    revenue = calculate_revenue(param.inventory_limit, param.unit_revenue, demand_dist)
    reward = {}
    for inv_level in range(param.inventory_limit + 1):
        for prod_level in range(min(param.inventory_limit - inv_level, param.production_limit) + 1):
            expected_revenue = revenue[inv_level + prod_level]
            fixed_cost = 0 if prod_level == 0 else param.setup_cost
            prod_cost = param.production_cost * prod_level
            inv_cost = param.inventory_cost * (inv_level + prod_level)
            reward[(inv_level, prod_level)] = expected_revenue - fixed_cost - prod_cost - inv_cost
    return reward


def calculate_transition_matrix(inv_limit: int, demand_dist: dict) -> list:
    transition_matrix = [[0 for _ in range(inv_limit + 1)] for _ in range(inv_limit + 1)]
    for level in range(inv_limit + 1):
        for demand, prob in demand_dist.items():
            transition_matrix[level][max(0, level - demand)] += prob
    return transition_matrix


def markov_decision_process(param: Param, demand_dist: dict) -> tuple:
    value_res = [[0 for _ in range(param.inventory_limit + 1)] for _ in range(param.time_horizon + 1)]
    action_res = [[0 for _ in range(param.inventory_limit + 1)] for _ in range(param.time_horizon + 1)]

    reward = calculate_reward(param, demand_dist)
    transition_matrix = calculate_transition_matrix(param.inventory_limit, demand_dist)

    for inv_level in range(param.inventory_limit + 1):
        value_res[param.time_horizon][inv_level] = inv_level * param.salvage_price

    for period in range(param.time_horizon - 1, -1, -1):
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


if __name__ == '__main__':
    instance = Instance()
    optimal_value, optimal_action = markov_decision_process(instance, demand_distribution)
    print(optimal_value)
    print(optimal_action)
