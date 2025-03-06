import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

# Add nodes (neurons)
layers = {0: 3, 1: 4, 2: 2}  

node_positions = {}
node_count = 0
for layer, num_nodes in layers.items():
    for i in range(num_nodes):
        G.add_node(node_count, layer=layer)
        node_positions[node_count] = (layer, -i)  
        node_count += 1

for src, data in G.nodes(data=True):
    for dst, dst_data in G.nodes(data=True):
        if dst_data["layer"] == data["layer"] + 1:  
            G.add_edge(src, dst)

plt.figure(figsize=(6, 6))
nx.draw(G, pos=node_positions, with_labels=True, node_size=700, node_color="lightblue")
plt.show()
