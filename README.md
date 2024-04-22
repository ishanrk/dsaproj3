# Brain Interface with Neuron Interaction

## What It Does:
The interface is used for visualizing how neurons interact in the brain if they were to be connected using algorithms such as dijkstra's, A*, and BFS for pathfinding between two neurons.
You can click on any two neurons and then click on the button for the corresponding algorithm to perform that algorithm and get the time and error accumulated in the console. Just as in
typical minimum distance problems, the weight we're trying to minimize here is the error accumulated over connectiosn which is determined by the neuron's error and connection_bias

## Data
The data is a json file called neuron_data.json that is greater than 150MB in size, thus it needs to be downloaded and put in the repository before running your program.
Link to dataset: https://drive.google.com/file/d/1Q2TIeIH032Jo5W-e3DntRArW1eZq5Ybt/view?usp=sharing
The data consists of fields: index, error, neuron_size, connection_bias, connections (adjacency list) and location used for visualization

## How to Run It
After you clone the repository to your device AND DOWNLOAD THE DATASET FROM THE LINK and put it in the same repository, then you should run interface.py.
Interface.py is the main driver for the program and uses pygame to visualize the interface
