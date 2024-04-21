import pygame
import random
import heapq
import math
import time
import algorithms
from hash import Hash
import graph

from interface_helper import generate_subset_neurons, neurons_to_graph, close_connections

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


#visualizes bfs on screen
def bfs_visualize(from_neuron, to_neuron, neurons, screen, bfs_button, generate_text2, num_neurons):
    
    #keeps track of visited neurns
    visited={}

    #queue list where elements removed from front and added at back
    queue_python_impl=[from_neuron]

    draw_edges=[] #keepsn track of edges to be drawn each instance

    #used for reconstructing path
    prev={}
    
    # animation only if neurons <=1000
    if num_neurons<=1000:

        #until destination found or all neurons traversed
        while(len(queue_python_impl)!=0):
            
            #get curr neuron
            curr_neuron = queue_python_impl.pop()
            #fill in visited
            visited[curr_neuron]=1
            
            #add neuron to list of adjacent edges to be drawn
            draw_edges.append(curr_neuron)

            #drawing bg
            screen.fill((230, 230, 255))
            screen.blit(BACK_IMAGE,(0,0))
            pygame.draw.rect(screen, RED, bfs_button)
            screen.blit(generate_text2, (930, 70))

            # select neuron for gray colour
            neurons[curr_neuron].selected=True
            
            #draw all neurons
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
                    pygame.draw.circle(screen, GRAY, (neuron.x_coord,neuron.y_coord), 8)
                
            # traverse adjacent edges
            for connection in neurons[curr_neuron].connections:

                if(connection not in visited.keys()):
                    queue_python_impl.append(connection)
                    prev[connection]=curr_neuron
                #if current neuron is destination then build a path and return
                if(connection==to_neuron):

                    path=[]
                    temp_neuron=connection
                    # build path by traveersing prev
                    while(temp_neuron!=from_neuron):
                        path.append(temp_neuron)
                        last_neuron=temp_neuron
                        #drawing
                        pygame.draw.circle(screen, RED, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord), 10)
                        temp_neuron=prev[temp_neuron]
                        pygame.draw.circle(screen, RED, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord), 10)
                        pygame.display.flip()
                        pygame.draw.line(screen, RED, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord),(neurons[last_neuron].x_coord,neurons[last_neuron].y_coord), 2)
                        #animation stop
                        time.sleep(0.05)
                    path.append(from_neuron)
                    pygame.display.flip()
                    screen.fill((230, 230, 255))
                    screen.blit(BACK_IMAGE,(0,0))
                    # deselect all neurons
                    for neuron in neurons:
                        if neurons[neuron].selected==True:
                            neurons[neuron].selected=False
                    return path
            # draw edges for each neuron processed
            for neuron in draw_edges:
            
                for connection in neurons[neuron].connections:
                
                    pygame.draw.line(screen, WHITE, (neurons[neuron].x_coord,neurons[neuron].y_coord), (neurons[connection].x_coord, neurons[connection].y_coord), 1)
        
            time.sleep(0.01)
            pygame.display.flip()
    else:
        # same as above but without visualizations
        screen.fill((230, 230, 255))
        screen.blit(BACK_IMAGE,(0,0))
        pygame.draw.rect(screen, RED, bfs_button)
        screen.blit(generate_text2, (930, 70))
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
        while(len(queue_python_impl)!=0):

            curr_neuron = queue_python_impl.pop()
            visited[curr_neuron]=1
            
            draw_edges.append(curr_neuron)
  
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
                        pygame.draw.circle(screen, GRAY, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord), 8)
                        temp_neuron=prev[temp_neuron]
                        pygame.draw.circle(screen, GRAY, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord), 8)
                        pygame.display.flip()
                        pygame.draw.line(screen, WHITE, (neurons[temp_neuron].x_coord,neurons[temp_neuron].y_coord),(neurons[last_neuron].x_coord,neurons[last_neuron].y_coord), 3)
                        
                        
                    path.append(from_neuron)
                    pygame.display.flip()
                    time.sleep(3)
                    screen.fill((230, 230, 255))
                    screen.blit(BACK_IMAGE,(0,0))
                    for neuron in neurons:
                        if neurons[neuron].selected==True:
                            neurons[neuron].selected=False
                    return path

        
       
        pygame.display.flip()

# bfs without visualization and gets error
def raw_bfs(from_neuron, to_neuron, neurons):
    visited={}

    #queue list where elements removed from front and added at back
    queue_python_impl=[from_neuron]

    #used for reconstructing path
    prev={}

    while(len(queue_python_impl)!=0):

            curr_neuron = queue_python_impl.pop()
            visited[curr_neuron]=1
  
            for connection in neurons[curr_neuron].connections:
                if(connection not in visited.keys()):
                    queue_python_impl.append(connection)
                    prev[connection]=curr_neuron
                if(connection==to_neuron):
                    path=[]
                    temp_neuron=connection
                    error_accumulate=0
                    while(temp_neuron!=from_neuron):
                        error_accumulate+=get_weight(neurons[prev[temp_neuron]], neurons[temp_neuron])
                        path.append(temp_neuron)
                        temp_neuron=prev[temp_neuron]
                        
                        
                    path.append(from_neuron)
               
                    return path, error_accumulate
    



# raw_astar is running the algo without visualization. Hueristic used is eucleadean dist between 2 neurons
# heuristic chosen because all neurons have connections to their 5 closest neurons making it similar to a grid
def raw_astar(from_neuron, to_neuron, neurons):
        
        #starting fscore
        fscore=((neurons[from_neuron].x_coord-neurons[to_neuron].x_coord)**2)+((neurons[from_neuron].y_coord-neurons[to_neuron].y_coord)**2)
        
        neuron_priority_q = [(fscore, from_neuron)]
        heapq.heapify(neuron_priority_q) #priority queue based on fscore

        #this visited dict tracks gscore along with checking if neuron is visited
        visited = {from_neuron: 0}
        # parentused for reconstructing path
        parent = {}

        #until destination neuron found or all neurons traversed
        while neuron_priority_q:
            #get high priority neuron based on fscore
            temp, current_neuron = heapq.heappop(neuron_priority_q) 
           

            if current_neuron == to_neuron:
                
                # get path
                path = []
                #build path through parent
                while current_neuron in parent:
                    
                    path.append(current_neuron)
                    current_neuron = parent[current_neuron]      
                path.append(from_neuron)
              
                return path[::-1]

            #traverse all edges
            for neighbor in neurons[current_neuron].connections:
                #gscore is added on each time by getting weight
                gscore = visited[current_neuron] + get_weight(neurons[current_neuron], neurons[neighbor])
                #if gscore smaller then add fscore based on euclidean distance(heuristic) onto priority queue and set new gscore in viisted
                if gscore < visited.get(neighbor, float('inf')):
                    parent[neighbor] = current_neuron
                    visited[neighbor] = gscore
                    fscore=gscore+ ((neurons[neighbor].x_coord-neurons[to_neuron].x_coord)**2)+((neurons[neighbor].y_coord-neurons[to_neuron].y_coord)**2)
                    heapq.heappush(neuron_priority_q, (fscore, neighbor)) 

            
# gets weight from connection bias, it represents the error accumulated between an edge a->b
def get_weight(a,b):
    return ((a.connection_bias)*(1-b.connection_bias))+((a.error+b.error))**2


# same as raw_astar but visualizes and gives track of error_accumulated in path
def astar_visualize(from_neuron, to_neuron, neurons, screen, astar_button, generate_text, num_neuron):

    # gets nodes still in queue
    fscore=((neurons[from_neuron].x_coord-neurons[to_neuron].x_coord)**2)+((neurons[from_neuron].y_coord-neurons[to_neuron].y_coord)**2)
    neuron_priority_q = [(fscore, from_neuron)]
    heapq.heapify(neuron_priority_q)

    # dict to track visited nodes and their costs
    visited = {from_neuron: 0}
    parent = {}
    if num_neuron<=1000:

        while neuron_priority_q:
            _, current_neuron = heapq.heappop(neuron_priority_q)
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

            if current_neuron == to_neuron:
                # Reconstruct path
                path = []
                
                last_neuron=0
                error_accumulate=0
                while current_neuron in parent:
                    error_accumulate+=get_weight(neurons[parent[current_neuron]],neurons[current_neuron])
                    path.append(current_neuron)
                    pygame.draw.circle(screen, RED, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), 10)
                    pygame.draw.line(screen, RED, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), (neurons[parent[current_neuron]].x_coord, neurons[parent[current_neuron]].y_coord), 5)
                    pygame.display.flip()
                    last_neuron=current_neuron
                    current_neuron = parent[current_neuron]
                    time.sleep(1)
                
                pygame.draw.circle(screen, RED, (neurons[from_neuron].x_coord,neurons[from_neuron].y_coord), 10)
                pygame.draw.line(screen, RED, (neurons[from_neuron].x_coord,neurons[from_neuron].y_coord), (neurons[path[::-1][0]].x_coord, neurons[path[::-1][0]].y_coord), 5)
                pygame.display.flip()
                time.sleep(5)
                path.append(from_neuron)
                
                return path[::-1], error_accumulate

            for neighbor in neurons[current_neuron].connections:
                pygame.draw.line(screen, GRAY, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), (neurons[neighbor].x_coord,neurons[neighbor].y_coord), 1)
                gscore = visited[current_neuron] + get_weight(neurons[neighbor],neurons[to_neuron])
                if gscore < visited.get(neighbor, float('inf')):
                    parent[neighbor] = current_neuron
                    visited[neighbor] = gscore
                    fscore=gscore+((neurons[from_neuron].x_coord-neurons[to_neuron].x_coord)**2)+((neurons[from_neuron].y_coord-neurons[to_neuron].y_coord)**2)

                    heapq.heappush(neuron_priority_q, (fscore, neighbor)) 
            time.sleep(0.001)
            pygame.display.flip()
    else:
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
        
        while neuron_priority_q:
            _, current_neuron = heapq.heappop(neuron_priority_q)
            

            if current_neuron == to_neuron:
                # Reconstruct path
                path = []
                error_accumulate=0
                last_neuron=0
                while current_neuron in parent:
                    error_accumulate+=get_weight(neurons[parent[current_neuron]],neurons[current_neuron])
                    path.append(current_neuron)
                    pygame.draw.circle(screen, GRAY, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), 8)
                    pygame.draw.line(screen, WHITE, (neurons[current_neuron].x_coord,neurons[current_neuron].y_coord), (neurons[parent[current_neuron]].x_coord, neurons[parent[current_neuron]].y_coord), 3)
                    pygame.display.flip()
                    last_neuron=current_neuron
                    current_neuron = parent[current_neuron]
                    time.sleep(0.01)
                pygame.draw.circle(screen, GRAY, (neurons[from_neuron].x_coord,neurons[from_neuron].y_coord), 8)
                pygame.draw.line(screen, WHITE, (neurons[from_neuron].x_coord,neurons[from_neuron].y_coord), (neurons[path[::-1][0]].x_coord, neurons[path[::-1][0]].y_coord), 3)
                pygame.display.flip()
                time.sleep(5)
                path.append(from_neuron)
              
                return path[::-1],error_accumulate

            for neighbor in neurons[current_neuron].connections:
                
                gscore = visited[current_neuron] +  get_weight(neurons[neighbor], neurons[to_neuron])
                if gscore < visited.get(neighbor, float('inf')):
                    parent[neighbor] = current_neuron
                    visited[neighbor] = gscore
                    fscore=gscore+((neurons[from_neuron].x_coord-neurons[to_neuron].x_coord)**2)+((neurons[from_neuron].y_coord-neurons[to_neuron].y_coord)**2)
                    heapq.heappush(neuron_priority_q, (fscore, neighbor)) 

# visualizes dijkstras
def dijk_visualize(from_neuron, to_neuron, graphBuild, screen, neurons):
    #gets time based on djikstra implementation from algorithms.py

    start_time=time.perf_counter_ns()
    dijk_path = algorithms.getPathNonContinous(from_neuron,to_neuron, graphBuild)
    end_time = time.perf_counter_ns()
    #get time needed to run

    error_accumulate=0
    # check error accumulation and visualize path
    x=0
    if(len(dijk_path)==1):
        return [],0,0
    while(x<len(dijk_path)-1):
        error_accumulate+=get_weight(neurons[dijk_path[x]], neurons[dijk_path[x+1]])
        pygame.draw.circle(screen, GRAY, (neurons[dijk_path[x]].x_coord,neurons[dijk_path[x]].y_coord), 8)
        pygame.draw.circle(screen, GRAY, (neurons[dijk_path[x+1]].x_coord,neurons[dijk_path[x+1]].y_coord), 8)
        
        pygame.draw.line(screen, WHITE, (neurons[dijk_path[x+1]].x_coord,neurons[dijk_path[x+1]].y_coord),(neurons[dijk_path[x]].x_coord,neurons[dijk_path[x]].y_coord), 3)
        pygame.display.flip()
        time.sleep(0.01)
        x+=1
   
    time.sleep(3)

    return dijk_path, end_time-start_time, error_accumulate


#main driver

def main():
    num_neurons = int(input("Enter number of neurons: ")) #gets num neurons max is 100k
    #generate a subset of neurons
    neurons = generate_subset_neurons(num_neurons)
    #pos dict maps neurons to position
    pos_dict={}
    sentence = input("Enter your NAME: ") #gets your name for funtionality
    
    # assigns neuron position based on neuron location in brain
    #divides image into 4 parts and uses lines to make sure neurons don't get placed outside of their respetive locations on the actual brain image
    #for eg. in the occipital lobe, it randomly assigns a position and makes sure it doesn't go out of bounds through a line representing
    # the lobe's brain boundary in the actual image. This is done for all neurons in 4 regions
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

    #after positions assigned, if neurons less than 5000 then add 5 closest connections to them.
    if(num_neurons<5000):
        close_connections(neurons)

    #get a graph obj from the neurons
    new_graph = neurons_to_graph(neurons)

    # set positons of all neurons in pos_dict
    for neuron in neurons.values():
        pos_dict[neuron.index] = (neuron.x_coord,neuron.y_coord)
    
    #begin visualization
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((1000, 834 ))
    pygame.display.set_caption("Brain Interface")
    font = pygame.font.SysFont(None, 24)
    font2 = pygame.font.SysFont(None, 48)
    

    #varaibales indicating algo chosen
    astar_chosen = False
    bfs_chosen=False
    edge_display=False
    name_func=False
    djik_chosen=False

    #buttons
    astar_button = pygame.Rect(50, 50, 100 , 70)
    generate_text = font.render("A Star", True, WHITE)

    bfs_button = pygame.Rect(900, 50, 100 , 70)
    generate_text2 = font.render("BFS", True, WHITE)

    edge_button = pygame.Rect(50, 150, 100 , 70)
    generate_text3 = font.render("Edge", True, WHITE)

    name_button = pygame.Rect(900, 150, 100 , 70)
    generate_text4 = font.render("Enter Name", True, BLACK)

    djik_button = pygame.Rect(150, 50, 100 , 70)
    generate_text5 = font.render("Djikstra", True, WHITE)
    
    brain_text = font2.render("Brain Interface", True, WHITE)
    
    visualization_on = True
    # source and destination neuron for pathfinding
    from_neuron=None
    to_neuron=None
    
    #run till program closed
    while visualization_on:
        # render bg, button, text etc
        screen.fill((230, 230, 255))
        screen.blit(BACK_IMAGE,(0,0))
        pygame.draw.rect(screen, GREEN, astar_button)
        screen.blit(brain_text, (400, 70))
        screen.blit(generate_text, (75, 80))

        pygame.draw.rect(screen, PINK, bfs_button)
        screen.blit(generate_text2, (932, 80))


        pygame.draw.rect(screen, RED, edge_button)
        screen.blit(generate_text3, (80, 175))

        pygame.draw.rect(screen, YELLOW, name_button)
        screen.blit(generate_text4, (905, 175))

        pygame.draw.rect(screen, GRAY, djik_button)
        screen.blit(generate_text5, (175, 80))

        #switch to check if any buttons pressed 
        for event in pygame.event.get():
            mouse_pos = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                visualization_on = False
            if event.type == pygame.MOUSEBUTTONDOWN and astar_button.collidepoint(mouse_pos) and from_neuron!=None and to_neuron!=None:
                astar_chosen=True
            if event.type == pygame.MOUSEBUTTONDOWN and bfs_button.collidepoint(mouse_pos) and from_neuron!=None and to_neuron!=None:
                bfs_chosen=True
            if event.type == pygame.MOUSEBUTTONDOWN and name_button.collidepoint(mouse_pos):
                name_func=True
            if event.type == pygame.MOUSEBUTTONDOWN and djik_button.collidepoint(mouse_pos) and from_neuron!=None and to_neuron!=None:
                djik_chosen=True
            if event.type == pygame.MOUSEBUTTONDOWN and edge_button.collidepoint(mouse_pos):
                if edge_display: 
                    edge_display=False
                else:
                    edge_display=True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: 
                # this specifically calculates which neuron was cicked and makes the selected filed tru and assigns it to either from_neuron or to_neuron
                for neuron in neurons.values():
                        distance = ((neuron.x_coord - mouse_pos[0]) ** 2 + (neuron.y_coord - mouse_pos[1]) ** 2) ** 0.5
                        if distance <= neuron.neuron_size*2:  
                            if neuron.selected: neuron.selected=False
                            else: neuron.selected=True
                            if from_neuron is None:
                                from_neuron= neuron.index
                                
                            elif to_neuron is None:
                                to_neuron = neuron.index
                               
                            break
        # calculated path
        path=[]  

        #if astar was chosen then get path and visualize it. Output time needed and error accumulated
        # error always>=dijkstra's optimal path
        if astar_chosen:
            path,error_astar=astar_visualize(from_neuron,to_neuron,neurons,screen, astar_button, generate_text,num_neurons)
            
            time.sleep(2)
            neurons[from_neuron].selected=False
            neurons[to_neuron].selected=False
            
            start_time = time.perf_counter_ns()
            raw_astar(from_neuron,to_neuron,neurons)
            end_time= time.perf_counter_ns()
            from_neuron=None
            to_neuron=None
            print("A STAR PATH: ",path)
            print("TIME TAKEN FOR A STAR IN NANOSECONDS: ", end_time-start_time)
            print("Error accumulated A star: ",error_astar)
            astar_chosen=False
            continue
        # same for bfs
        if bfs_chosen:
            path = bfs_visualize(from_neuron, to_neuron, neurons,screen, bfs_button, generate_text2,num_neurons)
            time.sleep(2)
            neurons[from_neuron].selected=False
            neurons[to_neuron].selected=False
            
            start_time=time.perf_counter_ns()
            temp,error_acc = raw_bfs(from_neuron,to_neuron,neurons)
            end_time= time.perf_counter_ns()
            
            bfs_chosen=False
            from_neuron=None
            to_neuron=None
            print("PATH TAKEN BY BFS IS: ", path)
            print("TIME TAKEN BY BFS IS: ", end_time-start_time)
            print("Error Accumulated by BFS is: ", error_acc)
            continue
        
        #render all neurons on screen
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
            if edge_display:
                #if edge displat then display edges of graph
                for neuron in neurons.values():
                    for connection in neuron.connections:
                        pygame.draw.line(screen, WHITE, (neuron.x_coord,neuron.y_coord), (neurons[connection].x_coord, neurons[connection].y_coord), 1)

        #entered name hashes to two neurons in the graph which are then connected through astar
        if name_func:
            currentHasher = Hash(num_neurons)  #hash obj for the sentence
            currentHasher.hash(sentence)

            path_taken=[]

            for word in currentHasher.map:
                temp_itr=0
                max_itr=currentHasher.map[word]
                for neuron in neurons.values():
                    if(temp_itr<max_itr):
                        currentHasher.map[word]=neuron.index
                    else:
                        break
                    temp_itr+=1
            
                path_taken.append(currentHasher.map[word])
                print(word, " maps to neuron ", neurons[currentHasher.map[word]].index , " in the location ", neurons[currentHasher.map[word]].location)
                pygame.draw.circle(screen, GRAY, ( neurons[currentHasher.map[word]].x_coord, neurons[currentHasher.map[word]].y_coord), 10)
                pygame.display.flip()
                time.sleep(3)
                
            print("LOADING PATH... ")

            x=0
            while(x<len(path_taken)-1):
                curr_path = raw_astar(path_taken[x],path_taken[x+1],neurons)
                y=0
                while(y<len(curr_path)-1):
                    pygame.draw.circle(screen, GRAY, (neurons[curr_path[y]].x_coord, neurons[curr_path[y]].y_coord), 8)
                    pygame.draw.circle(screen, GRAY, (neurons[curr_path[y+1]].x_coord, neurons[curr_path[y+1]].y_coord), 8)
                    pygame.draw.line(screen, WHITE, (neurons[curr_path[y]].x_coord, neurons[curr_path[y]].y_coord),  (neurons[curr_path[y+1]].x_coord, neurons[curr_path[y+1]].y_coord), 3)
                    time.sleep(0.05)
                    pygame.display.flip()
                    y+=1
                x+=1
                print("PATH TAKEN FOR PROCESSING YOUR NAME IN BRAIN: ", curr_path)
            pygame.display.flip()
            time.sleep(3)
           
            name_func=False
            continue
        #if dijkstra's chosen then visualize and output time and error
        if djik_chosen:
            path,time_taken,error_djik = dijk_visualize(from_neuron,to_neuron,new_graph,screen,neurons)

            print("Djikstra's shortest path: ", path)
            print("Time taken in nanoseconds: ", time_taken)
            print("Error Accumulated: ", error_djik)
            neurons[from_neuron].selected=False
            neurons[to_neuron].selected=False
            djik_chosen=False
            from_neuron=None
            to_neuron=None
            
            continue
            
        
        pygame.display.flip()
        clock.tick(60)
    #end
    pygame.quit()
    
#run main
if __name__ == "__main__":
    main()




