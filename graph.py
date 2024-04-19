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
        return "Node(index={}) -> {}".format(self.index, self.adjacencyList)


class Graph:
    def __init__(self, num_of_neurons : int = 100_000, num_of_regions: int = 8, max_connection_bias: float = 1.0, max_error : float = 0.2, max_adjacent : int = 500):
        self.num_of_neurons = num_of_neurons
        self.num_of_regions = num_of_regions
        self.max_connection_bias = max_connection_bias
        self.max_error = max_error
        self.max_adjacent = max_adjacent
        self._graph = ng.genNeuronsV3(num_of_neurons, num_of_regions, max_connection_bias, max_error, max_adjacent)

    def __getitem__(self, key : int):
        return Node(key, self._graph[0][key], self._graph[1][key], self._graph[2][key], self._graph[3][key])

    def getAdjacencyListOf(self, key):
        return self._graph[3][key]

    def getErrorBiasOf(self, key):
        return self._graph[2][key]

    def getConnectionBiasOf(self, key):
        return self._graph[1][key]

    def getRegionOf(self, key):
        return self._graph[0][key]

    def getAdjacencyList(self):
        return self._graph[3]

    def getNumberOfEdges(self):
        s = 0
        adj_list = self._graph[3]
        for i in range(adj_list.size):
            s += len(adj_list[i])
        return s

    def printAdjList(self):
        for i in range(self.num_of_neurons):
            print(self.__getitem__(i))

    def __repr__(self):
        return "Graph(num_of_neurons={}, num_of_regions={}, max_connection_bias={}, max_error={}, max_adjacent={})\nTotal Edges: {}".format(self.num_of_neurons, self.num_of_regions, self.max_connection_bias, self.max_error, self.max_adjacent, self.getNumberOfEdges())


class DNode:
    def __init__(self, index, distance):
        self.index = index
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __repr__(self):
        return "DNode(index={}, distance={})".format(self.index, self.distance)
