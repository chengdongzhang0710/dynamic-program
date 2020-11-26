import numpy as np

# policy choice
policies = {
    'policy_1': {'s_1': 'a_11', 's_2': 'a_21', 's_3': 'a_31'},
    'policy_2': {'s_1': 'a_11', 's_2': 'a_21', 's_3': 'a_32'},
    'policy_3': {'s_1': 'a_12', 's_2': 'a_21', 's_3': 'a_31'},
    'policy_4': {'s_1': 'a_12', 's_2': 'a_21', 's_3': 'a_32'},
}

# parameter
discount = 0.95
sample_path = 500
epoch = 500

reward = {'s_1': 1, 's_2': -1, 's_3': -2}

transition_prob = {
    'a_11': {'s_1': 0.5, 's_2': 0.25, 's_3': 0.25},
    'a_12': {'s_1': 0, 's_2': 0.75, 's_3': 0.25},
    'a_21': {'s_1': 0.25, 's_2': 0, 's_3': 0.75},
    'a_31': {'s_1': 0.25, 's_2': 0, 's_3': 0.75},
    'a_32': {'s_1': 0, 's_2': 0.5, 's_3': 0.5},
}

# markov chain sample
for index, policy in policies.items():
    total_sample_reward = 0
    for _ in range(sample_path):
        discount_reward = 0
        state = np.random.choice(list(reward.keys()), p=[0.25, 0.5, 0.25])
        for _ in range(epoch - 1):
            discount_reward = reward[state] + discount_reward * discount
            action = policy[state]
            state = np.random.choice(list(transition_prob[action].keys()), p=list(transition_prob[action].values()))
        total_sample_reward += discount_reward
    average_sample_reward = total_sample_reward / sample_path
    print(index, ': ', average_sample_reward)
    print('\n')
