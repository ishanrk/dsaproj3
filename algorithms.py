import numpy as np
import heapq
from graph import Graph, DNode
from collections import deque

def weightfunc(e1 : float, c1 : float, e2 : float, c2 : float):
    return (e1+e2)**2 + (c1*(1 - c2))

def dijkstra(source : int, graph : Graph, weightfunc = weightfunc):
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

def getPath(source : int, dest :int, graph : Graph, weightfunc = weightfunc) -> list[int]:
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

def breadth_first_search(source: int, dest: int, graph: Graph) -> list[int]:
    visited_nodes = set()
    queue = deque([source, [source]])

    while queue:
        node, path = queue.popleft()
        if node == dest:
            return path
        if node not in visited_nodes:
            visited_nodes.add(node)
            for next_node in graph.getAdjacencyListOf(node):
                if next_node not in visited_nodes:
                    queue.append((next_node, path + [next_node]))
    return []
