"""
This file is used to create a pytorch lighning dataloader for our data. 

The data is composed of many 3D graph (meshes) stored in mulitple pt files.
Labels and full description are stored in a csv file.

The loading is done from torch_geometric and then converted to a pytorch lighning dataset with lighting.LinghtningDataset
"""

# imports
import os
from typing import List, Tuple, Union
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset, InMemoryDataset
from torch_geometric.io import read_txt_array
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import NormalizeScale
from torch_geometric.utils import to_undirected
from torch_geometric.data import Batch
from itertools import repeat


# constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data

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




        
        

    