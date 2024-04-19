import unittest
import graph as g
import algorithms as a
import numpy as np

class TestDijkstra(unittest.TestCase):

    def test_valid_path(self):
        gr = g.Graph()
        p = a.getPath(np.random.randint(gr.num_of_neurons), np.random.randint(gr.num_of_neurons), gr)
        if len(p) > 1:
            for i in range(1, len(p)):
                self.assertTrue(p[i] in gr[p[i-1]].adjacencyList)
        else:
            self.assertTrue(p[0] == -1 or p[0] == -2)


if __name__ == "__main__":
    unittest.main()
