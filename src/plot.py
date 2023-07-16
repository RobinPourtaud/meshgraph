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


def plot3d(data):
    import plotly.graph_objs as go

    # Create a 3D scatter plot of nodes
    scatter = go.Scatter3d(
        x=data.x[:, 0],
        y=data.x[:, 1],
        z=data.x[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )

    # Create a list to hold the edge traces
    edge_traces = []

    # Add an edge trace for each pair of points
    for edge in data.edge_index.T:
        trace = go.Scatter3d(
            x=data.x[edge, 0],
            y=data.x[edge, 1],
            z=data.x[edge, 2],
            mode='lines',
            line=dict(
                color='rgb(125,125,125)',
                width=2
            )
        )
        edge_traces.append(trace)

    # Combine the scatter and edge traces, and plot them
    layout = go.Layout(
        showlegend=False,
        autosize=False,
        width=800,
        height=600,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False)
        )
    )

    fig = go.Figure(data=[scatter] + edge_traces, layout=layout)
    return fig


def main():
    # execute only if run as a script
    from utils.dataset import load_data
    import wandb
    wandb.init(project="autoencoder")
    conf = {
        "data_path": "data/Shapenet/raw/",
    }

    data = load_data("ShapeNet", conf)
    print(len(data))
    print(data[0])
    plt = plot3d(data[0])
    wandb.log({"plot": plt})


if __name__ == "__main__":
    # execute only if run as a script
    main()