import torch
import networkx as nx

G = nx.erdos_renyi_graph(5, 0.5, directed=True)

adj_matrix = nx.to_numpy_array(G)

adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
print(adj_tensor)
