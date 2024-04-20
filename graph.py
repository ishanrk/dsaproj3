import data.neuron_gen as ng
import json

class Node:
    def __init__(self, index : int, region : int, neuron_size : int, connection_bias : float, error_bias : float, adjacency_list):
        self.index = index
        self.region = region
        self.neuron_size = neuron_size
        self.connection_bias = connection_bias
        self.error_bias = error_bias
        self.adjacency_list = adjacency_list

    def __repr__(self):
        return "Node(index={}, region={}, neuron_size={}, connection_bias={}, error_bias={}, adjacencyList={})".format(self.index, self.region, self.neuron_size, self.connection_bias, self.error_bias, self.adjacency_list)

    def __str__(self):
        return "Node(index={}) -> {}".format(self.index, self.adjacency_list)

class GraphFromJson:
    """
    Graph where nodes are NOT indexed continuously. This means graph needs to be stored 
    as a dictionary. Needs to be build from JSON. Format of JSON must be:

        {"<index : int>" : {"<region : str>": int, "<neuron_size>" : int, "<connection_bias>" : float, "<error_bias>" : float, "<adjacency_list>" : list[int]}

         - Where <> refers to the name of the key in JSON followed by its type.

        Example:

        {"123" : {"region": 3, "neuron_size" : 1, "connection_bias" : 0.126, "error_bias" : 0.02, "adjacency_list" : [4, 2, 0]}

         If the specified format not provided, provide a dictionary that maps the standard specified above
         to the format used in JSON.

         Example:

         property_mapper = {"neuron_size" : "size", "adjacency_list" : "connections"}

    """
    def __init__(self, file, property_mapper = None):
        self.property_mapper = property_mapper
        with open(file, "r") as f:
            self._graph = json.load(f)
        self.num_of_neurons = len(self._graph)

    def __getitem__(self, key : int):
        node = self._graph[str(key)]
        return Node(key, node[self.mapper("region")], node[self.mapper("neuron_size")], node[self.mapper("connection_bias")], node[self.mapper("error_bias")], node[self.mapper("adjacency_list")])
    
    def __iter__(self):
         return iter(self._graph)

    def __len__(self):
         return len(self._graph)

    def mapper(self, property : str):
         """Maps user property to standard property specified in JSON format."""
         return self.property_mapper[property] if self.property_mapper else property



    def getAdjacencyListOf(self, key):
        node = self._graph[str(key)]
        return node[self.mapper("adjacency_list")]

    def getErrorBiasOf(self, key):
        node = self._graph[str(key)]
        return node[self.mapper("error_bias")]

    def getConnectionBiasOf(self, key):
        node = self._graph[str(key)]
        return node[self.mapper("connection_bias")]

    def getRegionOf(self, key):
        node = self._graph[str(key)]
        return node[self.mapper("region")]

    def getAdjacencyList(self):
        return {int(x) : self._graph[x][self.mapper("adjacency_list")] for x in self._graph}

    def getNumberOfEdges(self):
        s = 0
        adj_list = self.getAdjacencyList()
        for i in adj_list:
            s += len(adj_list[i])
        return s

    def printAdjList(self):
        for i in self._graph:
            print(self.__getitem__(i))

    def toJson(self, fileName):
        with open(fileName, 'w') as f:
            json.dump(self._graph, f)

    def __repr__(self):
        return "Graph(num_of_neurons={})\nTotal Edges: {}".format(self.num_of_neurons, self.getNumberOfEdges())


class Graph:
    """
    Graph where nodes are indexed continuously. This means that Nodes are created with 
    indexed ranging from 0 to num_of_neurons - 1. By being indexed continuously, graph
    can be stored compactly as a list of lists, where the corresponding values per 
    node are accessed in each list according to index.
    """
    def __init__(self, num_of_neurons : int = 100_000, num_of_regions: int = 8, max_neuron_size : int = 3, max_connection_bias: float = 1.0, max_error : float = 0.2, max_adjacent : int = 500):
        self.num_of_neurons = num_of_neurons
        self.num_of_regions = num_of_regions
        self.max_neuron_size = max_neuron_size
        self.max_connection_bias = max_connection_bias
        self.max_error = max_error
        self.max_adjacent = max_adjacent
        self._graph = ng.genNeuronsV3(num_of_neurons, num_of_regions, max_neuron_size, max_connection_bias, max_error, max_adjacent)

    def __getitem__(self, key : int):
        # TODO gen neuron_size.
        return Node(key, self._graph[0][key], self._graph[1][key], self._graph[2][key], self._graph[3][key], self._graph[4][key])

    def getAdjacencyListOf(self, key):
        return self._graph[4][key]

    def getErrorBiasOf(self, key):
        return self._graph[3][key]

    def getConnectionBiasOf(self, key):
        return self._graph[2][key]

    def getRegionOf(self, key):
        return self._graph[0][key]

    def getAdjacencyList(self):
        return self._graph[4]

    def getNumberOfEdges(self):
        s = 0
        adj_list = self._graph[4]
        for i in range(adj_list.size):
            s += len(adj_list[i])
        return s

    def printAdjList(self):
        for i in range(self.num_of_neurons):
            print(self.__getitem__(i))

    def toJson(self, fileName):

        with open(fileName, 'w') as f:
            l = self._graph.tolist();
            for i in range(len(l)):
                l[i] = l[i].tolist()
            for i in range(len(l[4])):
                l[4][i] = l[4][i].tolist();

            j = {'Graph' : l}
            json.dump(j, f)

    def __repr__(self):
        return "Graph(num_of_neurons={}, num_of_regions={}, max_neuron_size={}, max_connection_bias={}, max_error={}, max_adjacent={})\nTotal Edges: {}".format(self.num_of_neurons, self.num_of_regions, self.max_neuron_size, self.max_connection_bias, self.max_error, self.max_adjacent, self.getNumberOfEdges())


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
