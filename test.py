import networkx as nx
import matplotlib.pyplot as plt

G = nx.random_graphs.gnm_random_graph(10, 20)

print(list(G.nodes().keys()))

pos = {
    0: (0, 0),
    1: (1, 0),
    2: (0, 1),
    3: (1, 1),
    4: (0, 2),
    5: (2, 0),
    6: (2, 2),
    7: (2, 1),
    8: (1, 2),
    9: (10,0),
}
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', linewidths=0.1, font_size=10)
plt.show()