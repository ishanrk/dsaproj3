import numpy as np
import heapq
from graph import Graph, DNode

<<<<<<< HEAD
def weightfunc(e1 : float, c1 : float, e2 : float, c2 : float):
    return (e1+e2)**2 + (c1*(1 - c2))

def dijkstra(source : int, graph : Graph, weightfunc = weightfunc):
=======
def dijkstra(source : int, graph : Graph, weightfunc):
>>>>>>> d18c59d380265144ce78057d3414e8e20c9c1ce9
    """
    Computes shortest path from source to all reachable nodes.

    Returns a tuple containing two numpy.array.

    The first array represents the predecessor of node n in the shortest path,
    where n is the index of the node.

    The second array represents the shortest distance from source to node n. 
    """

    nset = np.zeros(graph.num_of_neurons, dtype=bool)
    ndist = np.full(graph.num_of_neurons, float("inf"))
    nprev = np.full(graph.num_of_neurons, -1)

    pq : list[DNode] = []

    # Going to detect path not found if dist -1. Prev can stay to -1. Or -2 to indicate source.
    nset[source] = True;
    ndist[source] = 0;
    nprev[source] = -2;

    heapq.heappush(pq, DNode(index=source, distance=ndist))

    while(len(pq) > 0):
        """
        This is the revision step, we get the item with smallest distance. Compare if it can 
        update adjacent:

            dist[curr.index] + weightfunc < dist[other].
        """
        curr = heapq.heappop(pq)

        nset[curr.index] = True # Item added is only when it is from the one the search is performed.

        for adj_idx in graph.getAdjacencyListOf(curr.index):

            if adj_idx == curr.index:
                continue

            e1 = graph.getErrorBiasOf(curr.index)
            c1 = graph.getConnectionBiasOf(curr.index)
            e2 = graph.getErrorBiasOf(adj_idx)
            c2 = graph.getConnectionBiasOf(adj_idx)

            abs_dist = ndist[curr.index] + weightfunc(e1, c1, e2, c2)
            
            # All altered distance iems are pushed to the queue.
            if abs_dist < ndist[adj_idx]:
                ndist[adj_idx] = abs_dist
                nprev[adj_idx] = curr.index
                if not nset[adj_idx]:
                    heapq.heappush(pq, DNode(index=adj_idx, distance=abs_dist))

    return (nprev, ndist)

<<<<<<< HEAD
def getPath(source : int, dest :int, graph : Graph, weightfunc) -> list[int]:
=======
def getPath(source : int, dest :int, graph : Graph, weightfunc = lambda a, b, c, d : abs(a+b+c+d)) -> list[int]:
>>>>>>> d18c59d380265144ce78057d3414e8e20c9c1ce9
    """
    Computes shortest path from source to destination. Under the hood, this function
    uses Dijkstra's shortest path algorithim. If you plan to compute multiple shortest 
    paths from a given node n, use the dijkstra method in this same package instead.
    """
    (prev, _) = dijkstra(source, graph, weightfunc)
    if(prev[dest] == -1):
        return [-1]
    elif(prev[dest] == -2):
        return [-2]
    else:
        path = [dest, prev[dest]]
        while prev[path[len(path) - 1]] != -2:
            path.append(prev[path[len(path) - 1]])

        return path[::-1]

