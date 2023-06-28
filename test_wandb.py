import wandb
from torch import optim, nn, utils, Tensor, cat
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torch_geometric import datasets
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import to_networkx




class VariationalGraphAutoencoder(pl.LightningModule):


    def __init__




