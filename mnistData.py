from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import InMemoryDataset
import torch
import wandb
import time

wandb.init(project="Mesh-Learning")

class FilteredMNISTSuperpixels(InMemoryDataset):
    """
    FilteredMNISTSUperpixels update every graph by: 
    - removing edges if the minimum "x" value of the two nodes is less than 0.6
    - remove "x" value
    - rename "pos" to "x"

    Furthermore, it ads a vizualization method to plot using networkx and matplotlib the data (nbGraph as parameter)
    """
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(FilteredMNISTSuperpixels, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some.file']

    def len(self):
        return len(self.data)
    
    def get(self, idx):
        data, slices = self.data, self.slices
        return data
    
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        dataset = MNISTSuperpixels(self.root)

        data_list = []
        for i in range(len(dataset)):
            wandb.log({"progress": i/len(dataset)})
            # log time to finish one object
            
            start = time.time()
            data = dataset[i]
            edge_index = data.edge_index
            edges_to_keep = []

            # Filter edges: 'x' value of both nodes >= 0.6
            for column in range(edge_index.size(1)):
                if data.x[edge_index[0, column]] >= 0.6 and data.x[edge_index[1, column]] >= 0.6:
                    edges_to_keep.append(column)
                    data.pos[edge_index[0, column]] = torch.empty(2)
                    data.pos[edge_index[1, column]] = torch.empty(2)

            data.edge_index = data.edge_index[:, edges_to_keep]

            

            # Rename "pos" to "x"
            data.x = data.pos
            data.pos = None

            data_list.append(data)
            end = time.time()
            wandb.log({"time": end-start})

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def visualize(self, idGraph, log=False):
        """
        Visualize the graph with idGraph

        Args:
            idGraph (int or range): id of the graph to visualize
            log (bool, optional): Log the graph to wandb. Defaults to False.

        """
        import networkx as nx
        import plotly.express as px
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt


        isRange = isinstance(idGraph, range)

        if type(idGraph) == int:
            idGraph = [idGraph]

        for nbGraph in idGraph:
            # Get the specific graph
            data = self[nbGraph]

            # Convert to NetworkX for visualization
            G = nx.from_edgelist(data.edge_index.numpy().T)

            # Get the position of each node
            pos = data.x.numpy()

            # Get the label of each node
            label = data.y.numpy()

            # Get title of the plot 
            title = "Number: " + str(data.y)

            # Get node degrees to size nodes by their degree
            degrees = [d for n, d in G.degree()]

            # Get a color map to color nodes by their label
            cmap = plt.get_cmap('viridis')
            

            # Plot using plotly
            fig = px.scatter(x=pos[:, 0], y=pos[:, 1], title=title, size=degrees)
            fig.update_traces(marker=dict(size=20, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))            

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                fig.add_trace(
                    go.Scatter(x=tuple([x0, x1, None]), y=tuple([y0, y1, None]),
                                mode='lines',
                                line=dict(color='rgb(210,210,210)', width=1),
                                hoverinfo='none'
                                )
                )
            fig.show()


            # Log to wandb
            if log:
                wandb.log({str(nbGraph): fig})

        



# Load the dataset
dataset = FilteredMNISTSuperpixels(root='data/MNIST')

# Visualize the first graph
dataset.visualize(range(7), log=True)
