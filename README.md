# Learning NetworkX: Graphs, CNNs, GNNs & More

## Overview
This repository documents my journey in learning NetworkX, an essential Python library for graph-based computations. I explore various applications, including visualizing graphs of Convolutional Neural Networks (CNNs), Graph Neural Networks (GNNs), and more.

## Why NetworkX?
NetworkX is a powerful tool for:
- Creating and manipulating graphs
- Analyzing network structures
- Visualizing relationships in deep learning models

## Getting Started
### Installation
Ensure you have Python installed, then install NetworkX:
```bash
pip install networkx matplotlib
```

### Basic Usage
```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 1)])

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()
```

## Applications Explored
- **Graph Visualization**: Understanding how CNN and GNN architectures can be represented as graphs.
- **Model Structure Analysis**: Investigating connections in deep learning architectures.
- **Custom Graphs**: Creating and analyzing networks for AI/ML research.

## Future Goals
- Implement more advanced graph algorithms.
- Explore real-world applications in AI and ML.
- Optimize graph visualizations for large-scale models.


