
import numpy as np
import pandas as pd

def get_neurons(neuron_number):
    location = ["temporal_lobe", "frontal_lobe", "occipital_lobe", "parietal_lobe"]
    neurons = []
    for x in range(0,neuron_number):
        
        specific_location = np.random.choice(location) # WHICH QUANDRANT OF THE SCREEN WILL THE NEURON BE TOP RIGHT.. TOP LEFT ETC.

        connection_bias = np.random.rand()
        error = np.random.uniform(0, 0.2)  # ERROR VARIABLE NEEDED FOR WEIGHT CALCULATION BETWEEN TWO NEURONS
        index = x  # UNIQUE INDEX FOR EACH NEURON, THIS IS HOW WE INTENTIFY UNIQUE VERTICES

        neuron_size = np.random.randint(1,4) # INDICATES HOW BIGTHE NEURON WILL BE ON THE SCREEN FROM 1-3 SIZES 

        connection_num = np.random.randint(1, 50) # LISTS THE NUMBER OF CONNECTION EACH NEURON HAS
        
        # Generate random connections for each neuron, excluding self-loop
        connections = np.random.choice(np.delete(np.arange(neuron_number), x), #DELETES DUPLICATES AND GENERATES A LIST OF CONNECTIONS
                                        size=connection_num, replace=False).tolist() # GETS LIST OF SPECIFIC CONNECTIONS 
        
        neurons.append(({"index":index, "location": specific_location, 
                                         "connection_bias": connection_bias,
                                         "error": error, "neuron_size" : neuron_size, "connections": connections})) #GET ENTIRE 1-100,000 LIST OF NEURONS
      
    return neurons
        

# ALL FUNCTIONS WILOL BE WRITTEN WITH INDEX OF NEURON AS PARAMATERS NOT THE ACTUAL NEURON OBJECT FOR SIMPLICITY
#THIS FUNCTION TAKES IN FROM INDEX AND TO INDEX FOR NEURONS AND CALCULATES WEIGHT, METHOD OF WEIGHT CALCULATION DONE IS IRRELEVANT BUT BIOLOGICALLY ACCURATE
def get_weight(from_neuron, to_neuron, neurons): 
    weight = 0
    weight+=(1-from_neuron["connection_bias"])
    weight += from_neuron["error"]
    
    location = ["temporal_lobe", "frontal_lobe", "occipital_lobe", "parietal_lobe"]
    from_location=from_neuron["location"]
    to_location=to_neuron["location"]

    num1,num2=0,0

    for x in range(4):
        if location[x]==from_location:
            num1=x
        if location[x]==to_location:
            num2=x
    weight += (abs(num2-num1))
    weight*=10
    weight = round(weight)
    return weight

import heapq
def heuristic_cost(neuron1, neuron2, neurons):
    # Calculate the Euclidean distance between neurons as heuristic cost
    return get_weight(neuron1,neuron2,neurons)

def get_neighbors(neuron, neurons):
    # Retrieve neighboring neurons based on connections
    return [neurons[i] for i in neuron["connections"]]

def astar(start_index, goal_index, neurons):
    start_neuron = neurons[start_index]
    goal_neuron = neurons[goal_index]

    # Priority queue to store open nodes
    open_nodes = [(0, start_index)]
    heapq.heapify(open_nodes)

    # Dict to track visited nodes and their costs
    visited = {start_index: 0}
    parent = {}

    while open_nodes:
        _, current_neuron = heapq.heappop(open_nodes)

        if current_neuron == goal_index:
            # Reconstruct path
            path = []
            while current_neuron in parent:
                path.append(current_neuron)
                current_neuron = parent[current_neuron]
            path.append(start_index)
            return path[::-1]

        for neighbor in get_neighbors(neurons[current_neuron], neurons):
            tentative_g_score = visited[current_neuron] + 1
            if tentative_g_score < visited.get(neighbor["index"], float('inf')):
                parent[neighbor["index"]] = current_neuron
                visited[neighbor["index"]] = tentative_g_score
                f_score = tentative_g_score + heuristic_cost(neighbor, goal_neuron,neurons)
                heapq.heappush(open_nodes, (f_score, neighbor["index"]))

    return None  # No path found

def row_to_neuron(row):
    return {
        "index": row["index"],
        "location": row["location"],
        "connection_bias": row["connection_bias"],
        "error": row["error"],
        "neuron_size": row["neuron_size"],
        "connections": row["connections"]
    }

# Convert DataFrame to a list of neurons




def main():
    neuron_dataframe = pd.read_json('data.json') # THIS IS HOW YOU GET ALL THE DATA IN YOUR PYTHON PROGRAM

    neurons = [row_to_neuron(row) for _, row in neuron_dataframe.iterrows()]

# Print the first few neurons
    for neuron in neurons[:5]:
        print(neuron)

    print(astar(0,10009,neurons)) # finds path between 0 and 10009, you can try yourself for any values
    # we need to do this for 1000 neurons


# Display the DataFrame
   
  



if __name__ == "__main__":
    main()