
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import networkx as nx
import networkx.algorithms.community as nx_comm
from sklearn.metrics.pairwise import cosine_similarity


def create_network_graph(word_vectors, threshold=0.5):
    """
    Create a 2D network graph given a dictionary of words and their associated word vectors.
    The graph is created using the networkx library and visualized using the matplotlib library.

    Args:
        1. words (dict): A dictionary where keys are words and values are their corresponding vectors.
        2. threshold (float): The threshold for cosine similarity to consider an edge between two nodes.

    The Function:
        1. 

    Returns:
        None
    """

    def get_word_vector(word):
        try:
            return word_vectors[word]
        except KeyError:
            return None
    
    def calculate_similarity(word1, word2):
        vector_1 = get_word_vector(word1)
        vector_2 = get_word_vector(word2)
        if vector_1 is None or vector_2 is None:
            return None
        similarity = cosine_similarity(vector_1.reshape(1, -1), vector_2.reshape(1, -1))[0][0]
        return similarity
        
    def create_word_graph(min_similarity=-1.0):
        graph = nx.Graph()
        words = list(word_vectors)
    
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                similarity = calculate_similarity(words[i], words[j])
                if similarity is not None and similarity > min_similarity:
                    graph.add_edge(words[i], words[j], weight=similarity)
        return graph
    
    word_graph = create_word_graph(threshold)

    if len(word_graph.edges) == 0:
        print("No edges were created. Check the threshold or word vectors.")
        return

    # Generate 2D positions for nodes
    pos = nx.spring_layout(word_graph)

    # Extract edge weights for coloring
    edge_weights = [d['weight'] for (u, v, d) in word_graph.edges(data=True)]
    if not edge_weights:
        print("Edge weights list is empty. No edges in the graph.")
        return

    norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    edge_colors = [cm.viridis(norm(weight)) for weight in edge_weights]

    # Detect communities (clusters)
    communities = nx_comm.greedy_modularity_communities(word_graph)
    community_colors = cm.tab10(np.linspace(0, 1, len(communities)))

    # Assign colors to nodes based on their community
    node_colors = {}
    for i, community in enumerate(communities):
        for node in community:
            node_colors[node] = community_colors[i]

    # Draw the graph with community-based node colors
    nx.draw(
        word_graph, pos, with_labels=True, node_size=800,
        node_color=[node_colors[node] for node in word_graph.nodes()],
        font_size=10, font_weight="bold", edge_color=edge_colors, edge_cmap=cm.viridis,
        width=[weight * 5 for weight in edge_weights]
    )

    # Add edge labels
    # edge_labels = nx.get_edge_attributes(word_graph, 'weight')
    # nx.draw_networkx_edge_labels(word_graph, pos, edge_labels=edge_labels)

    # Create a ScalarMappable for the colorbar
    # sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    # sm.set_array(edge_weights)

    # Add the colorbar
    # plt.colorbar(sm, label="Similarity")

    plt.title("(Work in progress)\n3D Network Graph of Word Vectors", fontsize=16)
    
    # Save the graph as an image
    plt.savefig("network_graph_2d.png", format="PNG")
    plt.show()


if __name__ == "__main__":

    data_file='data/project_data/training_data/test/project_word_vectors_over_time_01.npy'
    trained_word_vectors = np.load(data_file, allow_pickle=True)
    trained_word_vectors = trained_word_vectors[-1]
    create_network_graph(trained_word_vectors, threshold=0.05)
