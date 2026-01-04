import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools


def create_network_graph(word_vectors, threshold=0.5):
    """
    Create a 3D network graph given a dictionary of words and their associated word vectors.
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

    # Generate 3D positions for nodes
    pos = nx.spring_layout(word_graph, dim=3)

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

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract 3D positions
    node_xyz = np.array([pos[node] for node in word_graph.nodes()])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in word_graph.edges()])

    # Draw nodes
    ax.scatter(
        node_xyz[:, 0], node_xyz[:, 1], node_xyz[:, 2],
        c=[node_colors[node] for node in word_graph.nodes()],
        s=800, edgecolors="k", depthshade=True
    )

    # Draw edges
    for edge in edge_xyz:
        ax.plot(
            edge[:, 0], edge[:, 1], edge[:, 2],
            color="gray", alpha=0.5, linewidth=1
        )

    # Add labels to nodes
    for node, (x, y, z) in zip(word_graph.nodes(), node_xyz):
        ax.text(x, y, z, node, fontsize=10, fontweight="bold")

    # Add colorbar for edge weights
    # sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    # sm.set_array(edge_weights)
    # fig.colorbar(sm, ax=ax, label="Similarity")

    plt.title("(Work in progress)\n3D Network Graph of Word Vectors", fontsize=16)

    # Save the graph as an image
    plt.savefig("network_graph_3d.png", format="PNG")
    plt.show()


if __name__ == "__main__":
    word_vectors_over_time = np.load("word_vectors_over_time_02.npy", allow_pickle=True)

    trained_word_vectors = word_vectors_over_time[-1]

    create_network_graph(trained_word_vectors, threshold=0.05)