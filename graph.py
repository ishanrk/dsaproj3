import data.neuron_gen as ng

class Node:
    def __init__(self, index : int, region : int, connection_bias : float, error_bias : float, adjacencyList):
        self.index = index
        self.region = region
        self.connection_bias = connection_bias
        self.error_bias = error_bias
        self.adjacencyList = adjacencyList

    def __repr__(self):
        return "Node(index={}, region={}, connection_bias={}, error_bias={}, adjacencyList={})".format(self.index, self.region, self.connection_bias, self.error_bias, self.adjacencyList)

    def __str__(self):
        return "Node(index={}, region={}, connection_bias={}, error_bias={}, len(adjacencyList) = {})".format(self.index, self.region, self.connection_bias, self.error_bias, len(self.adjacencyList))



class Graph:
    def __init__(self, num_of_neurons : int = 100_000, num_of_regions: int = 8, max_connection_bias: float = 1.0, max_error : float = 0.2, max_adjacent : int = 500):
        self._graph = ng.genNeuronsV3(num_of_neurons, num_of_regions, max_connection_bias, max_error, max_adjacent)

    def __getitem__(self, key : int):
        return Node(key, self._graph[0][key], self._graph[1][key], self._graph[2][key], self._graph[3][key])

