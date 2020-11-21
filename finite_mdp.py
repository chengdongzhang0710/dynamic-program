# data input
prob_1 = [[0.9, 0.1, 0], [0.5, 0.4, 0.1], [0, 0.55, 0.45]]
prob_2 = [[0.8, 0.2, 0], [0.7, 0.1, 0.2], [0, 0.9, 0.1]]
prob = {'m1': prob_1, 'm2': prob_2}

expected_cost = {
    ('L1', 'm1'): 2,
    ('L1', 'm2'): 4,
    ('L2', 'm1'): 6,
    ('L2', 'm2'): 8,
    ('L3', 'm1'): 10,
    ('L3', 'm2'): 12,
}

disease_severity = ['L1', 'L2', 'L3']
medication_dose = ['m1', 'm2']

planning_horizon = 7

# finite markov decision process algorithm
value_func = [[0 for level in disease_severity] for t in range(planning_horizon + 1)]
actions = [['no action' for level in disease_severity] for t in range(planning_horizon)]

for t in range(planning_horizon - 1, -1, -1):
    for level_index, level in enumerate(disease_severity):
        dose_cost = {}
        for dose in medication_dose:
            cost = expected_cost[(level, dose)]
            for i in range(len(disease_severity)):
                cost += prob[dose][level_index][i] * value_func[t + 1][i]
            dose_cost[dose] = cost
        optimal_action = min(dose_cost, key=dose_cost.get)
        value_func[t][level_index] = dose_cost[optimal_action]
        actions[t][level_index] = optimal_action

# result output
print(value_func)
print(actions)
