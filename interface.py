import pygame
import random
import heapq
import math
import time
import pandas as pd
from neuron_gen import genNeuronsV3

# get colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK=(255,192,203)
RED=(255,0,0)


BACK_IMAGE= pygame.image.load("brain2.png") # brain image as bg


# gets data from neuron_gen, saves a local copy
def get_data():
    neurons = genNeuronsV3(num_of_neurons=100000, num_of_regions=4) # gets data from neuron_gen

    indices = range(0,100000)
    location = neurons[0].to_list()
    connection_bias = neurons[1].to_list()
    error = neurons[2].to_list()
    connections=[]
    for connection in neurons[3]:
        connections.append(connection.to_list())

    neuron_size = [random.randint(1,3) for x in range(100000)]
    neurons_dataframe= pd.DataFrame({'index': indices, 'location':location,'connection_bias':connection_bias, 'error':error,'neruon_size':neuron_size})
    neurons_dataframe.to_json('data.json')

# get dijkstra's time


# function to convert df into set of rows
def row_to_neuron(row):
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
    neurons = [row_to_neuron(row) for _, row in neuron_dataframe.iterrows()]

    # uses set to store subset neurons
    in_graph=[]
    in_graph = set(in_graph)
    first_neuron = neurons[random.randint(0,100000)]
    in_graph.add(first_neuron["index"])

    current_Subset_lenght = 1
    random.seed(422) #random seed, hint:submission date!

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

def breadth_first_search(from_neuron, to_neuron, neurons, screen, bfs_button, generate_text2):

    visited={}

    queue_python_impl=[from_neuron]

    draw_edges=[]
    prev={}
    while(len(queue_python_impl)!=0):

        curr_neuron = queue_python_impl.pop()
        visited[curr_neuron]=1
        
        draw_edges.append(curr_neuron)
        screen.fill((230, 230, 255))
        screen.blit(BACK_IMAGE,(0,0))
        pygame.draw.rect(screen, RED, bfs_button)
        screen.blit(generate_text2, (930, 70))

        neurons[curr_neuron].selected=True
        
        for neuron in neurons.values():
            
            if neuron.selected==False:
                if(neuron.location=="occipital_lobe"):
                    pygame.draw.circle(screen, YELLOW, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="temporal_lobe"):
                    pygame.draw.circle(screen, GREEN, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="parietal_lobe"):
                    pygame.draw.circle(screen, PINK, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="frontal_lobe"):
                    pygame.draw.circle(screen, RED, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
            else:
                pygame.draw.circle(screen, GRAY, (neuron.x_coord,neuron.y_coord), 10)
            
        print("this happen")
        
        for connection in neurons[curr_neuron].connections:
            if(connection not in visited.keys()):
                queue_python_impl.append(connection)
                prev[connection]=curr_neuron
            if(connection==to_neuron):
                path=[]
                temp_neuron=connection
                
                while(temp_neuron!=from_neuron):
                    path.append(temp_neuron)
                    last_neuron=temp_neuron
                    pygame.draw.circle(screen, RED, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord), 10)
                    temp_neuron=prev[temp_neuron]
                    pygame.draw.circle(screen, RED, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord), 10)
                    pygame.display.flip()
                    pygame.draw.line(screen, RED, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord),(neurons[last_neuron].x_coord,neurons[last_neuron].y_coord), 5)
                    
                    time.sleep(0.05)
                path.append(from_neuron)
                pygame.display.flip()
                screen.fill((230, 230, 255))
                screen.blit(BACK_IMAGE,(0,0))
                for neuron in neurons:
                    if neurons[neuron].selected==True:
                        neurons[neuron].selected=False
                return path

       
                        
            
        for neuron in draw_edges:
            print(neuron)
            for connection in neurons[neuron].connections:
                
                pygame.draw.line(screen, WHITE, (neurons[neuron].x_coord,neurons[neuron].y_coord), (neurons[connection].x_coord, neurons[connection].y_coord), 1)
        
        time.sleep(0.01)
        pygame.display.flip()
        

def close_connections(nodes):
    for node in nodes.values():
        # calc eucleadean distances to all other nodes
        distances = []
        for other_node in nodes.values():
            if other_node != node:
                distance = math.sqrt((other_node.x_coord - node.x_coord) ** 2 + (other_node.y_coord - node.y_coord) ** 2)
                distances.append((other_node.index, distance))
        
        # get lowest 2 distances and then add them to connections
        distances.sort(key=lambda x: x[1])
        for neighbor, _ in distances[:2]:
            node.connections.append(neighbor)

# gets weight from connection bias, lower weight=lower speed to traverse
def get_weight(a,b):
    return 100*((1-a.connection_bias)*(1-b.connection_bias))


# currently heurisitc is 0
def astar(start_index, goal_index, neurons, screen, astar_button, generate_text):

    # gets nodes still in queue
    open_nodes = [(0, start_index)]
    heapq.heapify(open_nodes)

    # dict to track visited nodes and their costs
    visited = {start_index: 0}
    parent = {}

    while open_nodes:
        _, current_neuron = heapq.heappop(open_nodes)
        screen.fill((230, 230, 255))
        screen.blit(BACK_IMAGE,(0,0))
        pygame.draw.rect(screen, RED, astar_button)
        screen.blit(generate_text, (80, 70))
        for neuron in neurons.values():
            
            if neuron.selected==False:
                if(neuron.location=="occipital_lobe"):
                    pygame.draw.circle(screen, YELLOW, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="temporal_lobe"):
                    pygame.draw.circle(screen, GREEN, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="parietal_lobe"):
                    pygame.draw.circle(screen, PINK, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="frontal_lobe"):
                    pygame.draw.circle(screen, RED, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
            else:
                pygame.draw.circle(screen, GRAY, (neuron.x_coord,neuron.y_coord), 10)
        pygame.draw.circle(screen, GRAY, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), 10)

        if current_neuron == goal_index:
            # Reconstruct path
            path = []
            
            last_neuron=0
            while current_neuron in parent:
                print("happen")
                path.append(current_neuron)
                pygame.draw.circle(screen, RED, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), 10)
                pygame.draw.line(screen, RED, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), (neurons[parent[current_neuron]].x_coord, neurons[parent[current_neuron]].y_coord), 5)
                pygame.display.flip()
                last_neuron=current_neuron
                current_neuron = parent[current_neuron]
                time.sleep(1)
            pygame.draw.circle(screen, RED, (neurons[start_index].x_coord,neurons[start_index].y_coord), 10)
            pygame.draw.line(screen, RED, (neurons[start_index].x_coord,neurons[start_index].y_coord), (neurons[path[::-1][0]].x_coord, neurons[path[::-1][0]].y_coord), 5)
            time.sleep(5)
            path.append(start_index)
            print(path[::-1])
            return path[::-1]

        for neighbor in neurons[current_neuron].connections:
            pygame.draw.line(screen, GRAY, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), (neurons[neighbor].x_coord,neurons[neighbor].y_coord), 1)
            score = visited[current_neuron] + 1
            if score < visited.get(neighbor, float('inf')):
                parent[neighbor] = current_neuron
                visited[neighbor] = score
                f_score = score + get_weight(neurons[neighbor], neurons[goal_index])
                heapq.heappush(open_nodes, (f_score, neighbor)) 
        time.sleep(0.001)
        pygame.display.flip()


def main():
    neurons = generate_subset_neurons(500)
    pos_dict={}

    for neuron in neurons.values():
        if(neuron.location=="occipital_lobe"):
            xmin=70
            xmax=500
            ymin=500
            ymax=734

            x_loc = random.randint(xmin,xmax)
            y_loc = random.randint(ymin, math.floor(600+(134/600)*x_loc))
            pos_dict[neuron.index] = (x_loc,y_loc)
            neuron.set_x(x_loc)
            neuron.set_y(y_loc)
        if(neuron.location=="temporal_lobe"):
            xmin=70
            xmax=500
            ymin=200
            ymax=500
            
            x_loc = random.randint(xmin,xmax)
            
            y_loc= random.randint(math.floor(450+(-250/430)*x_loc),ymax)
            pos_dict[neuron.index] = (x_loc,y_loc)

            neuron.set_x(x_loc)
            neuron.set_y(y_loc) 
        if(neuron.location=="parietal_lobe"):
            xmin=500
            xmax=950
            ymin=500
            ymax=700
            
            x_loc = random.randint(xmin,xmax)
            
            y_loc= random.randint(ymin,math.floor(935+(-134/400)*x_loc))
            
            pos_dict[neuron.index] = (x_loc,y_loc)
            neuron.set_x(x_loc)
            neuron.set_y(y_loc)
        if(neuron.location=="frontal_lobe"):
            xmin=500
            xmax=950
            ymin=250
            ymax=500
            
            x_loc = random.randint(xmin,xmax)
            
            y_loc= random.randint(math.floor(-83.3+(1/2)*x_loc),ymax)
            
            pos_dict[neuron.index] = (x_loc,y_loc)
            neuron.set_x(x_loc)
            neuron.set_y(y_loc)

    close_connections(neurons)

    for neuron in neurons.values():
        pos_dict[neuron.index] = (neuron.x_coord,neuron.y_coord)
    
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((1000, 834 ))
    pygame.display.set_caption("Brain Interface")
    font = pygame.font.SysFont(None, 24)
    

    astar_chosen = False
    bfs_chosen=False

    astar_button = pygame.Rect(50, 50, 100 , 70)
    generate_text = font.render("A Star", True, WHITE)

    bfs_button = pygame.Rect(900, 50, 100 , 70)
    generate_text2 = font.render("BFS", True, WHITE)
    
        
    
    running = True
    start_index=None
    goal_index=None
    
    while running:
        screen.fill((230, 230, 255))
        screen.blit(BACK_IMAGE,(0,0))
        pygame.draw.rect(screen, GREEN, astar_button)
        screen.blit(generate_text, (80, 70))

        pygame.draw.rect(screen, PINK, bfs_button)
        screen.blit(generate_text2, (930, 70))

        for event in pygame.event.get():
            mouse_pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and astar_button.collidepoint(mouse_pos) and start_index!=None and goal_index!=None:
                astar_chosen=True
            if event.type == pygame.MOUSEBUTTONDOWN and bfs_button.collidepoint(mouse_pos) and start_index!=None and goal_index!=None:
                bfs_chosen=True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: 
                             
                for neuron in neurons.values():
                        distance = ((neuron.x_coord - mouse_pos[0]) ** 2 + (neuron.y_coord - mouse_pos[1]) ** 2) ** 0.5
                        if distance <= neuron.neuron_size*2:  
                            if neuron.selected: neuron.selected=False
                            else: neuron.selected=True
                            if start_index is None:
                                start_index = neuron.index
                                print(start_index)
                            elif goal_index is None:
                                goal_index = neuron.index
                                print(goal_index)
                            break
        path=[]  
        if astar_chosen:
            path=astar(start_index,goal_index,neurons,screen, astar_button, generate_text)
            #path = breadth_first_search(start_index, goal_index, neurons,screen)
            time.sleep(2)
            neurons[start_index].selected=False
            neurons[goal_index].selected=False
            start_index=None
            goal_index=None
            print("hm")
            astar_chosen=False
            continue
        if bfs_chosen:
            path = breadth_first_search(start_index, goal_index, neurons,screen, bfs_button, generate_text2)
            time.sleep(2)
            neurons[start_index].selected=False
            neurons[goal_index].selected=False
            start_index=None
            goal_index=None
            print("hm")
            bfs_chosen=False
            continue
            
        for neuron in neurons.values():
            
            if neuron.selected==False:
                if(neuron.location=="occipital_lobe"):
                    pygame.draw.circle(screen, YELLOW, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="temporal_lobe"):
                    pygame.draw.circle(screen, GREEN, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="parietal_lobe"):
                    pygame.draw.circle(screen, PINK, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                if(neuron.location=="frontal_lobe"):
                    pygame.draw.circle(screen, RED, (neuron.x_coord,neuron.y_coord), neuron.neuron_size)
                
            else:
                pygame.draw.circle(screen, GRAY, (neuron.x_coord,neuron.y_coord), neuron.neuron_size*2)

        
        
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    



#run
if __name__ == "__main__":
    main()
