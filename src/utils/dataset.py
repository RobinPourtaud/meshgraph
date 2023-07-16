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
    elif dataset == "Shapenet":
        return load_ShapeNet(conf)
    elif dataset == "Embedding":
        return load_Embedding(conf)
    else:
        raise NotImplementedError("The dataset {} is not implemented".format(dataset))
def load_Embedding(conf : dict) -> InMemoryDataset:
    import clip 
    from torch.utils.data import Dataset
    import random
    import re

    text = {
        0: ["A photo of a zero", "nil", "nonexistent", "0", "null", "nought", "zilch", "zeroth", "cipher", "naught"],
        1: ["A photo of the number one", "singular", "individual", "1", "initial", "unity", "sole", "primary", "uno", "first"],
        2: ["A snapshot of a two", "dual", "couple", "2", "pair", "double", "duo", "binary", "twice", "second"],
        3: ["A picture of a three", "triple", "triplet", "3", "trio", "third", "threesome", "tertiary", "trifecta", "trilogy"],
        4: ["An image of a four", "quadruple", "quartet", "4", "quad", "fourth", "tetrad", "quadruplet", "quartile", "quadrant"],
        5: ["A photo of a five", "quintuple", "quintet", "5", "pentad", "fifth", "quinary", "quint", "pentagon", "pentacle"],
        6: ["A shot of a six", "sextuple", "sextet", "6", "hexad", "sixth", "senary", "hexagon", "half a dozen", "hex"],
        7: ["A picture of a seven", "septuple", "septet", "7", "heptad", "seventh", "septenary", "heptagon", "seventh heaven", "week"],
        8: ["A photo of an eight", "octuple", "octet", "8", "octad", "eighth", "octonary", "octagon", "octave", "eightfold"],
        9: ["An image of a nine", "nonuple", "nonet", "9", "nonad", "ninth", "nonary", "ennead", "novem", "ninefold"]
    }
    
    embeddings_dict = {}
    list_files = os.listdir(conf["data_path"])
    for file in list_files:
        # Extract the key and digit using regular expressions
        key = int(re.search(r'(\d+)_tensor', file).group(1))
        digit = int(re.search(r'\[(\d+)\]', file).group(1))
        # Choose a random sentence from the corresponding list in the dictionary
        sentence = random.choice(text[digit])
        # Add to the dictionary with key as the first number
        embeddings_dict[key] = sentence

    # Create a list of sentences based on the key order
    list_txt = [embeddings_dict[key] for key in sorted(embeddings_dict.keys())]
    _, preprocess = clip.load("ViT-B/32",device= conf["device_clip"], jit=False)

    target_size = [16, 16]
    from torchvision import transforms
    from torch.functional import F
    from PIL import Image

    class graph_title_dataset(Dataset):
        def __init__(self, list_graph_path,list_txt, text):
            self.graph_path = [conf["data_path"] + "/" + file for file in list_graph_path]
            self.title = clip.tokenize(list_txt) 
            self.text = text

        def __len__(self):
            return len(self.title)

        def __getitem__(self, idx):
            image = torch.load(self.graph_path[idx])
            im = transforms.ToPILImage()(image).convert("RGB")
            image = preprocess(im)
            title = self.title[idx]
            return image, title
        
        def get_classes(self):
            l = []
            for txt in self.text.values():
                l.extend(txt)
            return l
        
        
        
    return graph_title_dataset(list_files,list_txt, text)
    
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




def load_ShapeNet(conf : dict) -> InMemoryDataset:
    import pandas as pd
    df = pd.read_csv('data/Shapenet/processed/completeToKeep.csv')
    validId = df.where(df['y'] == "Chair").dropna()['fullId']
    validFiles = [id + '.pt' for id in validId]
    data_list = [torch.load(os.path.join(conf["data_path"], file)) for file in validFiles]
    for data, id in zip(data_list, validId):
        # keep only the first 3 dimensions
        data.pos = data.x[:, :3]
        data.x = data.pos
        data.y = df.where(df['fullId'] == id).dropna()['yFull'].values[0]
    return data_list
