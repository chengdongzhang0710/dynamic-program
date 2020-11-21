def wagner_whitin(demand: list, fixed_cost: float, inventory_cost: float):
    n = len(demand)
    result = [0] * (n + 1)
    duration = [-1] * n
    cost = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n - i):
            cost[i][j] = fixed_cost
            for k in range(1, j + 1):
                cost[i][j] += inventory_cost * k * demand[i + k]
    for i in range(n - 1, -1, -1):
        value = [(cost[i][j] + result[i + j + 1]) for j in range(n - i)]
        result[i] = min(value)
        duration[i] = value.index(min(value))
    return result, duration


if __name__ == '__main__':
    # Demand
    d = [2, 6, 5, 7, 2, 5, 3]
    # Setup Cost
    K = 4
    # Holding Cost
    h = 1
    # Production Cost
    b = 2
    v, t = wagner_whitin(d, K, h)
    print("Value Function:", v)
    print("Cover Period:", t)
    print("Optimal Value:", v[0] + b * sum(d))
