import numpy as np

field_to_index = {}

# 5 properties, map each other in the resulting json.
# This way data is lighter weight. Is index even necessary?
field_to_index["index"] = 0
field_to_index["connection bias"] = 1
field_to_index["error"] = 2
field_to_index["location"] = 3
field_to_index["connections"] = 4


def genNeuronsV1(num_of_neurons : int, num_of_regions : int = 8, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 500, bench_mark : bool = False):
    """Generates dataset of neurons returns a list with data."""

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, max_adjacent if bench_mark else np.random.randint(max_adjacent), replace=False)

    return np.array([regions, connection_bias, error_bias, connections])

def genNeuronsV2(num_of_neurons : int, num_of_regions : int = 8, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 500, bench_mark : bool = False):
    """Generates dataset of neurons returns a tuple with data."""

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, max_adjacent if bench_mark else np.random.randint(max_adjacent), replace=False)

    return np.array((regions, connection_bias, error_bias, connections))

def genNeuronsV3(num_of_neurons : int, num_of_regions : int = 8, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 500, bench_mark : bool = False):
    """Generates dataset of neurons returns a np.array with data."""

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, max_adjacent if bench_mark else np.random.randint(max_adjacent), replace=False)

    ret = np.empty(4, dtype=object)

    ret[0] = regions
    ret[1] = connection_bias
    ret[2] = error_bias
    ret[3] = connections


    return ret
