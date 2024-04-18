import numpy as np

field_to_index = {}

# 5 properties, map each other in the resulting json.
# This way data is lighter weight. Is index even necessary?
field_to_index["index"] = 0
field_to_index["connection bias"] = 1
field_to_index["error"] = 2
field_to_index["location"] = 3
field_to_index["connections"] = 4


# Base W/ list
def genNeuronsV1(num_of_neurons : int, num_of_regions : int = 8, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 50):

    """Going to build multiple arrays, each will be matched according to index. Therefore, to get
    data for say node 10, youll just need to do regions[10], connectionBias[10]... etc. We could extract
    this into a wrapper, which will just abstract this and return an object.
    """

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, np.random.randint(max_adjacent))

    return np.array([regions, connection_bias, error_bias, connections])

# W/ tuple
def genNeuronsV2(num_of_neurons : int, num_of_regions : int = 8, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 50):

    """Going to build multiple arrays, each will be matched according to index. Therefore, to get
    data for say node 10, youll just need to do regions[10], connectionBias[10]... etc. We could extract
    this into a wrapper, which will just abstract this and return an object.
    """

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, np.random.randint(max_adjacent))

    return np.array((regions, connection_bias, error_bias, connections))

# W/ empty
def genNeuronsV3(num_of_neurons : int, num_of_regions : int = 8, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 50):

    """Going to build multiple arrays, each will be matched according to index. Therefore, to get
    data for say node 10, youll just need to do regions[10], connectionBias[10]... etc. We could extract
    this into a wrapper, which will just abstract this and return an object.
    """

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, np.random.randint(max_adjacent))

    ret = np.empty(4, dtype=object)

    ret[0] = regions
    ret[1] = connection_bias
    ret[2] = error_bias
    ret[3] = connections


    return ret
