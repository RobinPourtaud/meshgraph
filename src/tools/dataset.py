"""
This file is used to create a pytorch lighning dataloader for our data. 

The data is composed of many 3D graph (meshes) stored in mulitple pt files.
Labels and full description are stored in a csv file.

The loading is done from torch_geometric and then converted to a pytorch lighning dataset with lighting.LinghtningDataset
"""

# imports
from torch_geometric.data import InMemoryDataset
import torch

def load_MNISTSuperpixels(conf : dict) -> InMemoryDataset:
    """ Load the MNIST Superpixels dataset from torch_geometric

    Args:
        conf (dict): The configuration file

    Returns:
        InMemoryDataset: The dataset
    """
    print("Loading the data...")

    assert type(conf) is dict, "conf must be a dict"
    assert conf["data_path"] is not None, "data_path must be specified in the config file"
    from torch_geometric.datasets import MNISTSuperpixels
    from torch_geometric import transforms as T

    def pre_transform(data):
        """ This function is used to:
            - Remove the useless edges that are not connected to nodes with a value >= 0.65, nodes not useful to represent the number
            - Remove the useless nodes that are not connected to any edge
            - Set the node features to the node positions

        Args:
            data (Data): The data to transform

        Returns:
            Data: The transformed data
        """
        edge_index = data.edge_index
        edges_mask = (data.x[edge_index[0]] >= 0.65) & (data.x[edge_index[1]] >= 0.65)
        edges_to_keep = torch.nonzero(edges_mask, as_tuple=True)[0]
        data.edge_index = edge_index[:, edges_to_keep]
        data.x = data.pos
        data = T.RemoveIsolatedNodes()(data)
        return data

    return MNISTSuperpixels(conf["data_path"], True, pre_transform=pre_transform)




def load_ShapeNet(conf) -> InMemoryDataset:
        # imports
    import os

    import pandas as pd
    class ShapeNet(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(ShapeNet, self).__init__(root, transform, pre_transform)
            # Load descriptions from CSV
            self.description_df = pd.read_csv('data/processed/description.csv')
            # Create a mapping of categories to integers
            self.category_to_int = {category: i for i, category in enumerate(self.description_df['subCategory'].unique())}

            # Load descriptions from CSV
            

        @property
        def raw_file_names(self):
            # Get all file names in the directory with '.pt' extension
            all_files = [os.path.splitext(file)[0] for file in os.listdir(self.root) if file.endswith('.pt')]
            
            # Filter the list to only include files that are present in description_df
            valid_files = [file for file in all_files if file in self.description_df['fileName'].values]
            
            return valid_files

        @property
        def processed_file_names(self):
            return self.raw_file_names()

        def download(self):
            pass

        def process(self):
            pass

        def len(self):
            return len(self.raw_file_names)

        def get(self, idx):
            file = self.raw_file_names[idx]
            data = torch.load(os.path.join(self.root, file))
            
            # Get label from description_df based on filename
            label = self.description_df.loc[self.description_df['fileName'] == file, 'subCategory'].values[0]
            
            # Convert the label to an integer
            label = self.category_to_int[label]
            
            # Add the label to the data object
            data.y = torch.tensor([label])
            
            return data

        






