"""
Varational Autoencoder for graph data


"""



# torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# torch lightening
import pytorch_lightning as pl

from torch_geometric.transforms import NormalizeFeatures

# torch log 
from torch.utils.tensorboard import SummaryWriter

# torch geometric
from torch_geometric.nn import VGAE
from torch_geometric.datasets import MNISTSuperpixels
# encoder
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, InnerProductDecoder


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torch_geometric.loader import DataLoader


class VGAELightningModule(pl.LightningModule):
    def __init__(self, encoder, decoder, train_dataset, lr, batch_size):
        super(VGAELightningModule, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.lr = lr

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=10)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu = None,
                logstd = None):
        
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def kl_loss(self, z, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        z = self.encode(x, edge_index)
        loss = self.kl_loss(z, self.__mu__, self.__logstd__)
        self.log('train_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def test_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        z = self.encode(x, edge_index)
        loss = self.kl_loss(z, self.__mu__, self.__logstd__)
        self.log('test_loss', loss)
        return loss
    
    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
class GenericEncoder(nn.Module):
    def __init__(self, ConvLayer, in_channels, hidden_channels, out_channels):
        super(GenericEncoder, self).__init__()
        self.conv_shared = ConvLayer(in_channels, hidden_channels)
        self.conv_mu = ConvLayer(hidden_channels, out_channels)
        self.conv_logvar = ConvLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv_shared(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar
    

def main(): 
    # load data
    dataset = MNISTSuperpixels(root="data/MNIST", transform=NormalizeFeatures())
    print("Dataset loaded")
    subset = dataset

    # create model
    encodList = [GCNConv, GATConv, SAGEConv]

    for encod in encodList:
        print(f"Training {encod.__name__}")
        encoder = GenericEncoder(encod, dataset.num_features, 16, 16)
        model = VGAELightningModule(
            encoder=encoder,
            decoder=InnerProductDecoder(),
            train_dataset=subset, 
            lr=0.01,
            batch_size=32
        )

        # Set up TensorBoard Logger
        logger = TensorBoardLogger("lightning_logs", name=f"{encod.__name__}")

        # create trainer
        trainer = pl.Trainer(devices=3, max_epochs=10, logger=logger)

        # train the model
        trainer.fit(model)

        # save the model
        torch.save(model.state_dict(), f"models/{encod.__name__}.pt")
if __name__ == "__main__":
    main()



