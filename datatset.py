import pandas as pd
import numpy as np
def lol():
    # Define the possible brain parts
    brain_parts = ["temporal_lobe", "frontal_lobe", "occipital_lobe", "parietal_lobe"]

    # Generate random data for 100,000 neurons
    num_neurons = 100000

    # Generate random data for each attribute
    part_of_brain = np.random.choice(brain_parts, num_neurons)
    connection_bias = np.random.rand(num_neurons)
    error = np.random.uniform(0, 0.2, num_neurons)
    location_index = np.random.randint(0, 100000, num_neurons)

    # Generate random number of connections for each neuron (1-1000)
    num_connections = np.random.randint(1, 1001, num_neurons)

    # Generate random connections for each neuron
    connections = [np.random.randint(0, 100000, size=n) for n in num_connections]

    # Create a DataFrame to store the neurons
    neurons_df = pd.DataFrame({
        "part_of_brain": part_of_brain,
        "connection_bias": connection_bias,
        "error": error,
        "location_index": location_index,
        "connections": connections
    })

    # Display the DataFrame
    print(neurons_df)

import networkx as nx
import matplotlib.pyplot as plt

# Function to generate random neurons
def generate_neurons(num_neurons):
    # Define the possible brain parts
    brain_parts = ["temporal_lobe", "frontal_lobe", "occipital_lobe", "parietal_lobe"]
    
    neurons = []
    for idx in range(num_neurons):
        print(idx)
        part_of_brain = np.random.choice(brain_parts)
        connection_bias = np.random.rand()
        error = np.random.uniform(0, 0.2)
        location_index = idx  # Unique identifier for each neuron

        # Generate random number of connections for each neuron (1-1000)
        num_connections = np.random.randint(1, 500)
        
        # Generate random connections for each neuron, excluding self-loop
        connections = np.random.choice(np.delete(np.arange(num_neurons), idx), 
                                        size=num_connections, replace=False).tolist()
        
        neurons.append(({"location_index":location_index, "part_of_brain": part_of_brain, 
                                         "connection_bias": connection_bias,
                                         "error": error, "connections": connections}))
    return neurons

# Function to display the graph
def display_graph(neurons):
    G = nx.Graph()
    G.add_nodes_from(neurons)
    for location_index in neurons["location_index"]:
        print(neurons[location_index]["connections"])
        for connection in neurons[location_index]["connections"]:
            G.add_edge(location_index, connection)
    pos = nx.spring_layout(G)  # Layout algorithm for graph visualization
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=6)
    plt.show()

# Main function
def main():
    num_neurons = int(input("Enter the number of neurons to generate (1-100000): "))
    if not 1 <= num_neurons <= 100000:
        print("Invalid input! Please enter a number between 1 and 100000.")
        return
    
    neurons = generate_neurons(num_neurons)
    print("check1")
    neurons_df = pd.DataFrame(neurons, columns=["location_index", "part_of_brain", "connection_bias", "error", "connections"])
    print("check2")
    neurons_df.to_csv("neurons_data.csv", index=False)
    

# Display DataFrame
    print(neurons_df)   

if __name__ == "__main__":
    main()
