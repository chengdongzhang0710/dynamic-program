class Network:
    nodes = []
    nbr = {}
    cost = {}

    def __init__(self, file_path: str):
        with open(file_path) as file:
            n = int(next(file))
            self.nodes = list(range(n))
            for i in range(n):
                self.nbr[i] = []
                line = next(file).split()
                for k in range(0, len(line) - 2, 2):
                    j = int(line[k + 2])
                    self.nbr[i].append(j)
                    self.cost.update({(i, j): int(line[k + 3])})


def dijkstra(network: Network, start=0):
    d = {i: float('inf') for i in network.nodes}
    d[start] = 0
    pred = [-1 for i in network.nodes]
    sd = [float('inf') for i in network.nodes]
    for k in range(len(network.nodes)):
        u = min(d, key=d.get)
        sd[u] = d[u]
        d.pop(u)
        for v in network.nbr[u]:
            if v in d.keys():
                if d[v] > sd[u] + network.cost[(u, v)]:
                    d[v] = sd[u] + network.cost[(u, v)]
                    pred[v] = u
    return sd, pred


if __name__ == '__main__':
    network = Network('test-case.txt')
    sd, pred = dijkstra(network, 0)  # if the start node is U, then start = 7; if the start node is N, then start = 0
    print('Distance Label:', sd)
    print('Predecessor:', pred)
