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
    return plt


def main():
    # execute only if run as a script
    from utils.dataset import load_data
    import wandb
    wandb.init(project="autoencoder")
    conf = {
        "data_path": "data/Shapenet"
    }
    data = load_data("ShapeNet", conf)
    print(data)
    for i in range(10):
        wandb.log({"graph": wandb.Image(plot_graph(data[i]))})



if __name__ == "__main__":
    # execute only if run as a script
    main()