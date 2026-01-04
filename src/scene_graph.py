import networkx as nx
import matplotlib.pyplot as plt

from utils.polygon_tools import check_positioning, create_polygons

class SceneGraph:
    def __init__(self, elements_polygons, elements_labels):
        self.elements_polygons = elements_polygons
        self.elements_labels = elements_labels

        if len(self.elements_polygons) != len(self.elements_labels):        
               print(f"WARNING--Mismatch: {len(self.elements_polygons)} polygons and {len(self.elements_labels)} labels")
        
        if len(self.elements_polygons) > len(self.elements_labels):
            # Extend elements_labels with empty strings to match elements_polygons length
            self.elements_labels.extend([""] * (len(self.elements_polygons) - len(self.elements_labels)))
        

        self.graph = self.create_graph()

    def create_graph(self):
        _, _, relative_positions = check_positioning(self.elements_polygons)

        # Initialize a directed graph
        G = nx.DiGraph()

        for i in range(len(self.elements_polygons)):
            G.add_node(i, label=self.elements_labels[i])
            
        for i, j, position_x, position_y in relative_positions:
            if position_x == "on":
                G.add_edge(i, j, label=position_x)
            else:
                G.add_edge(i, j, label=f"{position_x}, {position_y}")

        return G

    def plot_graph(self):
        # Use centroid positions to create a layout dictionary for nodes
        # pos = {i: (polygons[i].centroid.x, polygons[i].centroid.y) for i in range(len(polygons))}
        # pos = nx.spring_layout(self.graph)
        pos = nx.kamada_kawai_layout(self.graph)
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        node_labels = nx.get_node_attributes(self.graph, 'label')


        plt.figure(figsize=(8, 6))
        nx.draw(self.graph, pos, with_labels=True, labels=node_labels, node_size=1000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='blue')
        # plt.title("Scene Graph")
        plt.show()

def compare_scene_graphs(graph1, graph2):
    differences = {
        "missing_nodes": [],
        "extra_nodes": [],
        "mismatched_node_labels": [],
        "missing_edges": [],
        "extra_edges": [],
        "mismatched_edge_labels": []
    }
    
    # Build label-to-node mappings for graph1 and graph2
    label_to_node1 = {data['label']: node for node, data in graph1.nodes(data=True)}
    label_to_node2 = {data['label']: node for node, data in graph2.nodes(data=True)}
    
    # Compare nodes by label
    labels1 = set(label_to_node1.keys())
    labels2 = set(label_to_node2.keys())
    
    differences["missing_nodes"] = list(labels1 - labels2)
    differences["extra_nodes"] = list(labels2 - labels1)

    common_labels = labels1 & labels2
    for label in common_labels:
        node1 = label_to_node1[label]
        node2 = label_to_node2[label]
        if graph1.nodes[node1].get("label") != graph2.nodes[node2].get("label"):
            differences["mismatched_node_labels"].append(
                (label, graph1.nodes[node1].get("label"), graph2.nodes[node2].get("label"))
            )
    
    # Compare edges by label
    edges1 = {(graph1.nodes[u]['label'], graph1.nodes[v]['label']): graph1.edges[u, v]['label'] for u, v in graph1.edges}
    edges2 = {(graph2.nodes[u]['label'], graph2.nodes[v]['label']): graph2.edges[u, v]['label'] for u, v in graph2.edges}

    edge_keys1 = set(edges1.keys())
    edge_keys2 = set(edges2.keys())

    differences["missing_edges"] = list(edge_keys1 - edge_keys2)
    differences["extra_edges"] = list(edge_keys2 - edge_keys1)

    for edge in edge_keys1 & edge_keys2:
        label1 = edges1[edge]
        label2 = edges2[edge]
        if label1 != label2:
            differences["mismatched_edge_labels"].append(
                (edge, label1, label2)
            )
    
    return differences

def plot_graph_differences(graph1, graph2, differences):
    # Create a combined graph to display both graphs and differences
    combined_graph = nx.DiGraph()
    
    # Add nodes from both graphs with color-coding for differences
    for node, data in graph1.nodes(data=True):
        color = 'lightcoral' if data['label'] in differences["missing_nodes"] else 'lightgreen'
        combined_graph.add_node(data['label'], color=color)
        
    for node, data in graph2.nodes(data=True):
        if data['label'] not in combined_graph:
            color = 'skyblue' if data['label'] in differences["extra_nodes"] else 'lightgreen'
            combined_graph.add_node(data['label'], color=color)
    
    # Add edges from both graphs, highlighting differences
    for (u, v, data) in graph1.edges(data=True):
        edge_label = f"{graph1.nodes[u]['label']} -> {graph1.nodes[v]['label']}"
        color = 'red' if edge_label in differences["missing_edges"] else 'black'
        combined_graph.add_edge(graph1.nodes[u]['label'], graph1.nodes[v]['label'], color=color, label=data['label'])

    for (u, v, data) in graph2.edges(data=True):
        edge_label = f"{graph2.nodes[u]['label']} -> {graph2.nodes[v]['label']}"
        if edge_label not in combined_graph.edges:
            color = 'blue' if edge_label in differences["extra_edges"] else 'black'
            combined_graph.add_edge(graph2.nodes[u]['label'], graph2.nodes[v]['label'], color=color, label=data['label'])
    
    # Draw the graph
    pos = nx.spring_layout(combined_graph)
    edge_colors = [data['color'] for _, _, data in combined_graph.edges(data=True)]
    node_colors = [data['color'] for _, data in combined_graph.nodes(data=True)]

    plt.figure(figsize=(12, 8))
    nx.draw(combined_graph, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, node_size=500, font_size=10, font_color="black")
    
    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in combined_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(combined_graph, pos, edge_labels=edge_labels, font_color='gray')
    
    # Add a legend
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=10, label='Missing Node'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Extra Node'),
            plt.Line2D([0], [0], color='red', lw=2, label='Missing Edge'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Extra Edge')
        ],
        loc="upper right"
    )
    
    plt.title("Scene Graph Comparison")
    plt.show()
