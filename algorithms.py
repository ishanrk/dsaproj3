import numpy as np
import heapq
from collections import deque
from graph import Graph, DNode, GraphFromJson

def weightfunc(e1 : float, c1 : float, e2 : float, c2 : float):
    return (e1+e2)**2 + (c1*(1 - c2))

def dijkstraNonContinous(source : int, graph : GraphFromJson, weightfunc = weightfunc):
    """
    Computes shortest path from source to all reachable nodes.

    Returns a tuple containing two numpy.array, and a map mapping non-continuous nodes to indices.

    The first array represents the predecessor of node n in the shortest path,
    where n is the index of the node.

    The second array represents the shortest distance from source to node n. 
    """

    assert(str(source) in graph), f"source node is not in graph: {source=}"

    # Only use when dealing with nset, ndist, and nprev.
    m = {int(v) : i for i, v in enumerate(graph)}

    nset = np.zeros(len(graph), dtype=bool)
    ndist = np.full(len(graph), float("inf"))
    nprev = np.full(len(graph), -1)


    pq : list[DNode] = []

    ndist[m[source]] = 0;
    nprev[m[source]] = -2;

    heapq.heappush(pq, DNode(index=source, distance=ndist[m[source]]))

    while(len(pq) > 0):
        """
        This is the revision step, we get the item with smallest distance. Compare if it can 
        update adjacent:

            dist[curr.index] + weightfunc < dist[other].
        """
        curr = heapq.heappop(pq)

        nset[m[curr.index]] = True # Item added is only when it is from the one the search is performed.

        for adj_idx in graph.getAdjacencyListOf(curr.index):

            # Might be the case node in adjacency list is not in graph.
            if adj_idx == curr.index or str(adj_idx) not in graph:
                continue

            e1 = graph.getErrorBiasOf(curr.index)
            c1 = graph.getConnectionBiasOf(curr.index)
            e2 = graph.getErrorBiasOf(adj_idx)
            c2 = graph.getConnectionBiasOf(adj_idx)

            abs_dist = ndist[m[curr.index]] + weightfunc(e1, c1, e2, c2)
            
            # All altered distance iems are pushed to the queue.
            if abs_dist < ndist[m[adj_idx]]:
                ndist[m[adj_idx]] = abs_dist
                nprev[m[adj_idx]] = curr.index
                if not nset[m[adj_idx]]:
                    heapq.heappush(pq, DNode(index=adj_idx, distance=abs_dist))

    return (nprev, ndist, m)


def dijkstra(source : int, graph : Graph, weightfunc = weightfunc):

    """
    Computes shortest path from source to all reachable nodes.

    Returns a tuple containing two numpy.array.

    The first array represents the predecessor of node n in the shortest path,
    where n is the index of the node.

    The second array represents the shortest distance from source to node n. 

    source : int -> index of source node.

    graph : Graph

    """

    assert(source < graph.num_of_neurons), f"source node is not in graph: {source=}"

    nset = np.zeros(graph.num_of_neurons, dtype=bool)
    ndist = np.full(graph.num_of_neurons, float("inf"))
    nprev = np.full(graph.num_of_neurons, -1)

    pq : list[DNode] = []

    # Going to detect path not found if dist -1. Prev can stay to -1. Or -2 to indicate source.
    ndist[source] = 0;
    nprev[source] = -2;

    heapq.heappush(pq, DNode(index=source, distance=ndist[source]))

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
            
            # All altered distance items are pushed to the queue.
            if abs_dist < ndist[adj_idx]:
                ndist[adj_idx] = abs_dist
                nprev[adj_idx] = curr.index
                if not nset[adj_idx]:
                    heapq.heappush(pq, DNode(index=adj_idx, distance=abs_dist))

    return (nprev, ndist)

def getPath(source : int, dest :int, graph : Graph, weightfunc = weightfunc) -> list[int]:
    """
    Computes shortest path from source to destination. Under the hood, this function
    uses Dijkstra's shortest path algorithm. If you plan to compute multiple shortest 
    paths from a given node n, use the Dijkstra's method in this same package instead.
    """

    (prev, _) = dijkstra(source, graph, weightfunc)
    if(prev[dest] == -1):
        return [-1]
    elif(prev[dest] == -2):
        return [-2]
    else:
        path = [dest, prev[dest]]
        # While indexing the last item in our current path to find the previous,
        # if the previous is not -2 (root) continue, if it is -2 stop and return.
        while prev[path[len(path) - 1]] != -2:
            path.append(prev[path[len(path) - 1]])

        return path[::-1]

def getPathNonContinous(source : int, dest :int, graph : GraphFromJson, weightfunc = weightfunc) -> list[int]:
    """
    Computes shortest path from source to destination for graphs with non-continuous indexing.
    Under the hood, this function uses Dijkstra's shortest path algorithm. If you plan to compute 
    multiple shortest paths from a given node n, use the Dijkstra's method in this same package instead.
    """

    (prev, _, m) = dijkstraNonContinous(source, graph, weightfunc)
    if(prev[m[dest]] == -1):
        return [-1]
    elif(prev[m[dest]] == -2):
        return [-2]
    else:
        path = [dest, prev[m[dest]]]
        # While indexing the last item in our current path to find the previous,
        # if the previous is not -2 (root) continue, if it is -2 stop and return.
        while prev[m[path[len(path) - 1]]] != -2:
            path.append(prev[m[path[len(path) - 1]]])

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
