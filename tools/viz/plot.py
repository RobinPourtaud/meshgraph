import plotly.graph_objects as go
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_3D_networkx(networkx_graph, title=None):
    """Plot a 3D representation of a networkx graph.

    Args:
        networkx_graph: A networkx graph object.
        title: (Optional) A string representing the title of the plot.

    Returns:
        A Plotly Figure object.
    """
    assert isinstance(networkx_graph, nx.Graph), "Input must be a networkx graph object"

    x_nodes = [networkx_graph.nodes[k]['coordinates'][0] for k in networkx_graph.nodes]
    y_nodes = [networkx_graph.nodes[k]['coordinates'][1] for k in networkx_graph.nodes]
    z_nodes = [networkx_graph.nodes[k]['coordinates'][2] for k in networkx_graph.nodes]

    x_edges = []
    y_edges = []
    z_edges = []

    for edge in networkx_graph.edges:
        x0, y0, z0 = networkx_graph.nodes[edge[0]]['coordinates']
        x1, y1, z1 = networkx_graph.nodes[edge[1]]['coordinates']
        x_edges.extend([x0, x1, None])
        y_edges.extend([y0, y1, None])
        z_edges.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none'
    )

    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            color='blue',
            size=6,
            line=dict(color='black', width=0.5)
        ),
        hoverinfo='text',
        text=[str(k) for k in networkx_graph.nodes]
    )

    layout = go.Layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        scene=dict(
            xaxis=dict(title='X', showgrid=False, zeroline=False),
            yaxis=dict(title='Y', showgrid=False, zeroline=False),
            zaxis=dict(title='Z', showgrid=False, zeroline=False)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    return fig




def plot_2D_networkx(networkx_graph, method='pca', title=None):
    """Plot a 2D representation of a 3D networkx graph using dimensionality reduction.

    Args:
        networkx_graph: A networkx graph object.
        method: (Optional) A string representing the dimensionality reduction method to use. Options are 'pca' or 'tsne'. Default is 'pca'.
        title: (Optional) A string representing the title of the plot.

    Returns:
        A Plotly Figure object.
    """
    assert isinstance(networkx_graph, nx.Graph), "Input must be a networkx graph object"
    assert method in ('pca', 'tsne'), "Invalid method argument"

    node_positions_3D = np.array([networkx_graph.nodes[k]['vertex'] for k in networkx_graph.nodes])

    if method == 'pca':
        pca = PCA(n_components=2)
        node_positions_2D = pca.fit_transform(node_positions_3D)
    else:  # method == 'tsne'
        tsne = TSNE(n_components=2)
        node_positions_2D = tsne.fit_transform(node_positions_3D)

    x_nodes = node_positions_2D[:, 0]
    y_nodes = node_positions_2D[:, 1]

    x_edges = []
    y_edges = []

    for edge in networkx_graph.edges:
        x0, y0 = node_positions_2D[edge[0]]
        x1, y1 = node_positions_2D[edge[1]]
        x_edges.extend([x0, x1, None])
        y_edges.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=x_edges, y=y_edges,
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none'
    )

    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers',
        marker=dict(
            color='blue',
            size=6,
            line=dict(color='black', width=0.5)
        ),
        hoverinfo='text',
        text=[str(k) for k in networkx_graph.nodes]
    )

    layout = go.Layout(
        title=dict(text=title, x=0.5),
        showlegend=False,
        xaxis=dict(title='Component 1', showgrid=False, zeroline=False),
        yaxis=dict(title='Component 2', showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    return fig
