import networkx as nx
import matplotlib.pyplot as plt
def plot_graph(data):
    # to networkx
    from torch_geometric.utils.convert import to_networkx
    G = to_networkx(data, to_undirected=True)

    pos = {
        i : data.x[i].numpy() for i in range(len(data.x))
    }

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=10, width=0.5)
    # Add title
    plt.title('Graph rep of {}'.format(data.y))
    plt.show()