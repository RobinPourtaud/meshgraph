"""
This file is used to create a pytorch lighning dataloader for our data. 

The data is composed of many 3D graph (meshes) stored in mulitple pt files.
Labels and full description are stored in a csv file.

The loading is done from torch_geometric and then converted to a pytorch lighning dataset with lighting.LinghtningDataset
"""

# imports
from torch_geometric.data import InMemoryDataset
import torch
import os
import torch
def load_data(dataset : str, conf : dict) -> InMemoryDataset:
    """ Load the dataset

    Args:
        dataset (str): The dataset to load
        conf (dict): The configuration file

    Returns:
        InMemoryDataset: The dataset
    """
    assert type(dataset) is str, "dataset must be a string"
    assert type(conf) is dict, "conf must be a dict"
    assert conf["data_path"] is not None, "data_path must be specified in the config file"

    print("Loading dataset {}".format(dataset))
    if dataset == "MNISTSuperpixels":
        return load_MNISTSuperpixels(conf)
    elif dataset == "ShapeNet":
        return load_ShapeNet(conf)
    else:
        raise NotImplementedError("The dataset {} is not implemented".format(dataset))
    
    
def load_MNISTSuperpixels(conf : dict) -> InMemoryDataset:
    """ Load the MNIST Superpixels dataset from torch_geometric

    Args:
        conf (dict): The configuration file

    Returns:
        InMemoryDataset: The dataset
    """

    assert type(conf) is dict, "conf must be a dict"
    assert conf["data_path"] is not None, "data_path must be specified in the config file"
    from torch_geometric.datasets import MNISTSuperpixels
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform


    def pre_transform_MST(data, threshold = 0.7):
        """
        This function is used to:
            - Remove all edges
            - Remove every node with a position value < 0.8 by default (Euclidean distance)
            - Compute the minimum spanning tree based on the euclidean distance between nodes (data.pos)

        Args:
            data (Data): The data to transform
            threshold (float, optional): The threshold to remove nodes. Defaults to 0.75.

        Returns:
            Data: The transformed data
        """
        # Remove all edges
        data.edge_index = torch.tensor([], dtype=torch.long)
        valid_indices = torch.where(data.x >= threshold)[0]
        # Filter out the nodes from data.pos using these indices
        data.x = data.pos[valid_indices]
        # Compute pairwise distance matrix
        dist_matrix = pdist(data.x.detach().numpy())
        dist_matrix_square = squareform(dist_matrix)
        mst = minimum_spanning_tree(dist_matrix_square)
        mst = torch.tensor(mst.toarray(), dtype=torch.float)
        data.edge_index = torch.stack(torch.nonzero(mst, as_tuple=True))
        
        return data
    
    def pre_transform_KNN(data, k = 3, threshold = 0.55):
        """
        This function is used to:
            - Remove all edges
            - Remove every node with a position value < 0.8 by default (Euclidean distance)
            - Compute the minimum spanning tree based on the euclidean distance between nodes (data.pos)

        Args:
            data (Data): The data to transform
            threshold (float, optional): The threshold to remove nodes. Defaults to 0.75.

        Returns:
            Data: The transformed data
        """
        # Remove all edges
        data.edge_index = torch.tensor([], dtype=torch.long)
        valid_indices = torch.where(data.x >= threshold)[0]
        # Filter out the nodes from data.pos using these indices
        data.x = data.pos[valid_indices]
        data.pos = data.x
        from sklearn.neighbors import kneighbors_graph
        try:
            A = kneighbors_graph(data.pos.detach().numpy(), k, mode='connectivity', include_self=False)
            A_dense = torch.tensor(A.toarray(), dtype=torch.float)
            # Then find the indices where the adjacency matrix is non-zero
            data.edge_index = (A_dense != 0).nonzero(as_tuple=False).t()
    
        except:
            dist_matrix = pdist(data.x.detach().numpy())
            dist_matrix_square = squareform(dist_matrix)
            mst = minimum_spanning_tree(dist_matrix_square)
            mst = torch.tensor(mst.toarray(), dtype=torch.float)
            data.edge_index = torch.stack(torch.nonzero(mst, as_tuple=True))
            
        return data
        



    if not os.path.exists(conf["data_path"] + "/raw/MNISTSuperpixels.zip"):
        print("Downloading the data...")
        os.system("wget https://data.pyg.org/datasets/MNISTSuperpixels.zip -P " + conf["data_path"] + "/raw")

    return MNISTSuperpixels(conf["data_path"], True, pre_transform=pre_transform_KNN)




def load_ShapeNet(conf) -> InMemoryDataset:
        # imports

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
        
    return ShapeNet(conf["data_path"])

        






