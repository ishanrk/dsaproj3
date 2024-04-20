import numpy as np

field_to_index = {}

field_to_index["regions"] = 0
field_to_index["neuron_size"] = 1
field_to_index["connection_bias"] = 2
field_to_index["error_bias"] = 3
field_to_index["connections"] = 4


def genNeuronsV1(num_of_neurons : int = 100_000, num_of_regions : int = 8, max_neuron_size : int = 3, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 500, replace : bool = True,bench_mark : bool = False):
    """
    Generates dataset of neurons returns a list with data.
        [0]: regions
        [1]: neuron_size.
        [2]: connection_bias.
        [3]: error_bias.
        [4]: connections.
    """


    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    neuron_size = np.random.randint(1, max_neuron_size + 1, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, max_adjacent if bench_mark else np.random.randint(max_adjacent), replace=replace)

    return np.array([regions, neuron_size, connection_bias, error_bias, connections])

def genNeuronsV2(num_of_neurons : int = 100_00, num_of_regions : int = 8, max_neuron_size : int = 3, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 500, replace : bool = True, bench_mark : bool = False):
    """
    Generates dataset of neurons returns a tuple with data.
        [0]: regions
        [1]: neuron_size.
        [2]: connection_bias.
        [3]: error_bias.
        [4]: connections.
    """

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    neuron_size = np.random.randint(1, max_neuron_size + 1, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, max_adjacent if bench_mark else np.random.randint(max_adjacent), replace=replace)

    return np.array((regions, neuron_size, connection_bias, error_bias, connections))

def genNeuronsV3(num_of_neurons : int = 100_000, num_of_regions : int = 8, max_neuron_size : int = 3, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 500, replace : bool = True, bench_mark : bool = False):
    """
    Generates dataset of neurons returns a np.array with data.
        [0]: regions
        [1]: neuron_size.
        [2]: connection_bias.
        [3]: error_bias.
        [4]: connections.
    """

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    neuron_size = np.random.randint(1, max_neuron_size + 1, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, max_adjacent if bench_mark else np.random.randint(max_adjacent), replace=replace)

    ret = np.empty(5, dtype=object)

    ret[0] = regions
    ret[1] = neuron_size
    ret[2] = connection_bias
    ret[3] = error_bias
    ret[4] = connections

    return ret

def genNeuronsV3_no_size(num_of_neurons : int = 100_000, num_of_regions : int = 8, max_connection_bias : float = 1.0, max_error : float = 0.2, max_adjacent : int = 500, replace : bool = True, bench_mark : bool = False):
    """
    Generates dataset of neurons returns a np.array with data.
        [0]: regions
        [1]: connection_bias.
        [2]: error_bias.
        [3]: connections.
    """

    regions = np.random.randint(0, num_of_regions, num_of_neurons)
    connection_bias = np.random.rand(num_of_neurons) * max_connection_bias
    error_bias = np.random.rand(num_of_neurons) * max_error
    connections = np.empty(num_of_neurons, dtype=object)

    choices = np.arange(num_of_neurons)

    for i in range(num_of_neurons):
        connections[i] = np.random.choice(choices, max_adjacent if bench_mark else np.random.randint(max_adjacent), replace=replace)

    ret = np.empty(4, dtype=object)

    ret[0] = regions
    ret[1] = connection_bias
    ret[2] = error_bias
    ret[3] = connections

    return ret
