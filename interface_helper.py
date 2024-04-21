import json
import random
from neuron_gen import genNeuronsV3
import graph
import pandas as pd
import math

YELLOW = (255, 255, 0)

#format for converting json to graph
fe_to_std_json = {"region" : "location", "neuron_size" : "neuron_size", "connection_bias" : "connection_bias", "error_bias" : "error", "adjacency_list" : "connections"}


#randomly gens data and saves a local copy. Only run once before actually running driver to get copy of data
def get_data():
    neurons = genNeuronsV3(num_of_neurons=100000, num_of_regions=4) # gets data from neuron_gen

    #processes to dataframe to convert to json
    indices = range(0,100000)
    location = neurons[0].to_list()
    connection_bias = neurons[1].to_list()
    error = neurons[2].to_list()
    connections=[]
    for connection in neurons[3]:
        connections.append(connection.to_list())

    neuron_size = [random.randint(1,3) for x in range(100000)]
    neurons_dataframe= pd.DataFrame({'index': indices, 'location':location,'connection_bias':connection_bias, 'error':error,'neuron_size':neuron_size})
    neurons_dataframe.to_json('data.json')

# function to convert df into set of rows
def get_neuron_row(row):
    return {
        "index": row["index"],
        "location": row["location"],
        "connection_bias": row["connection_bias"],
        "error": row["error"],
        "neuron_size": row["neuron_size"],
        "connections": row["connections"]
    }

#all attributes of the neuron onscreen
class Neuron:
    def __init__(self, neuron_dict):
        self.index = neuron_dict["index"]
        self.location= neuron_dict["location"]
        self.connection_bias = neuron_dict["connection_bias"]
        self.error=neuron_dict["error"]
        self.neuron_size=neuron_dict["neuron_size"]
        self.connections = []
        self.x_coord = 0
        self.y_coord = 0
        self.color=YELLOW
        self.selected=False

    def set_x(self, x):
        self.x_coord = x
    
    def set_y(self,y):
        self.y_coord=y
    
    def connection_add(self, neuron_index):
        self.connections.append(neuron_index)

# randomly samples subset_number of neurons from total dataset
def generate_subset_neurons(subset_number):
    # read data
    neuron_dataframe = pd.read_json('data.json')
    neurons = [get_neuron_row(row) for _, row in neuron_dataframe.iterrows()]

    # uses set to store subset neurons
    in_graph=[]
    in_graph = set(in_graph)
    first_neuron = neurons[random.randint(0,100000)]
    in_graph.add(first_neuron["index"])

    current_Subset_lenght = 1
   

    # generates subset 
    while(current_Subset_lenght!=subset_number):
        random_neuron = neurons[random.randint(0,99999)]
        if random_neuron["index"] not in in_graph:
            in_graph.add(random_neuron["index"])
            current_Subset_lenght+=1

    # neuron map
    neuron_list={}
    
    # goes thorugh each in_graph neuron to create a Neuron object
    in_graph = list(in_graph)
    for neuron in in_graph:
        num_edges = random.randint(1,2)
        x=0
        new_neuron = Neuron(neurons[neuron])
        while(x!=num_edges):
            random_neuron = in_graph[random.randint(0,len(in_graph)-1)]
            if(random_neuron!=neuron):
                x+=1     
                new_neuron.connection_add(random_neuron)
        neuron_list[new_neuron.index] = new_neuron
    
    
    return neuron_list

def neurons_to_graph(neurons):
    temp_dict={}
    for neuron in neurons.values():
        temp_dict[neuron.index] = {"location":neuron.location, "error": neuron.error, "connection_bias":neuron.connection_bias, "neuron_size":neuron.neuron_size, "connections":neuron.connections}

    with open("neuron_subset.json", "w") as write: 
        json.dump(temp_dict, write)
    new_graph = graph.GraphFromJson("neuron_subset.json",fe_to_std_json)

    return new_graph

#adds edges between neurons that are geometrically close
def close_connections(nodes):
    for node in nodes.values():
        # calc eucleadean distances to all other nodes
        distances = []
        for other_node in nodes.values():
            if other_node != node:
                distance = math.sqrt((other_node.x_coord - node.x_coord) ** 2 + (other_node.y_coord - node.y_coord) ** 2)
                distances.append((other_node.index, distance))
        
        # get lowest 5 distances and then add them to connections
        distances.sort(key=lambda x: x[1])
        for neighbor, temp in distances[:5]:
            node.connections.append(neighbor)
