from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import InMemoryDataset
import torch
import time

class FilteredMNISTSuperpixels(InMemoryDataset):
    """
    FilteredMNISTSUperpixels update every graph by: 
    - removing edges if the minimum "x" value of the two nodes is less than 0.6
    - remove "x" value
    - rename "pos" to "x"

    Furthermore, it ads a vizualization method to plot using networkx and matplotlib the data (nbGraph as parameter)
    """
    
    def __init__(self, root, transform=None, pre_transform=None, wandb=None):
        self.wandb = wandb
        super().__init__(root, transform, pre_transform)
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
        # Read data into huge `Data` list. or download 
        dataset = MNISTSuperpixels("data/MNISTSuperpixels")

        data_list = []
        for i in range(min(len(dataset), 6)):
            self.wandb.log({"progress": i/len(dataset)})
            # log time to finish one object
            
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

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def visualize(self, idGraph, log=True, show=True):
        """
        Visualize the graph with idGraph with matplotlib and networkx

        Args:
            idGraph (int or list): id(s) of the graph(s) to visualize
            log (bool, optional): Log the graph to wandb. Defaults to True.
            show (bool, optional): Show the graph. Defaults to True.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        if isinstance(idGraph, int):
            idGraph = [idGraph]
        
        fig, axs = plt.subplots(len(idGraph), 1, figsize=(10, len(idGraph) * 10))

        for i, graph_id in enumerate(idGraph):
            data = self.get(graph_id)

            # Create a networkx graph
            G = nx.Graph()
            G.add_edges_from(data.edge_index.T.tolist())
            pos = {node: data.x[node].tolist() for node in range(data.x.shape[0])}

            # Draw the networkx graph using the positions from 'x'
            nx.draw(G, pos, ax=axs[i])

            # If we want to log the graph to wandb
            if log:
                self.wandb.log({"graph_" + str(graph_id): self.wandb.Image(plt)})

        # If we want to show the graph
        if show:
            plt.show()


    
