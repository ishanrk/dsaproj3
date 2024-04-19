import numpy as np
import heapq
from graph import Graph, DNode

def dijkstra(source : int, graph : Graph, weightfunc):
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

def getPath(source : int, dest :int, graph : Graph):
    (prev, _) = dijkstra(source, graph, lambda a, b, c, d : abs(a+b+c+d))
    if(prev[dest] == -1):
        return -1
    elif(prev[dest] == -2):
        return -2
    else:
        path = [dest, prev[dest]]
        while prev[path[len(path) - 1]] != -2:
            path.append(prev[path[len(path) - 1]])

        return path[::-1]
