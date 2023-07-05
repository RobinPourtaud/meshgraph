import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric import transforms as T
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch_geometric.loader import DataLoader
import wandb

if __name__ == "__main__":
    wandb.init(project="autoencoder")


def pre_transform(data):
    edge_index = data.edge_index
    edges_mask = (data.x[edge_index[0]] >= 0.65) & (data.x[edge_index[1]] >= 0.65)
    edges_to_keep = torch.nonzero(edges_mask, as_tuple=True)[0]
    data.edge_index = edge_index[:, edges_to_keep]
    data.x = data.pos
    data = T.RemoveIsolatedNodes()(data)
    return data

class GenericEncoder(nn.Module):
    def __init__(self, ConvLayer, in_channels, hidden_channels, out_channels):
        super(GenericEncoder, self).__init__()
        self.conv_shared = ConvLayer(in_channels, hidden_channels)
        self.conv_mu = ConvLayer(hidden_channels, out_channels)
        self.conv_logvar = ConvLayer(hidden_channels, out_channels)
        self.latent_dim = out_channels

    def forward(self, x, edge_index):
        x = F.relu(self.conv_shared(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar
    
class VGAELightning(pl.LightningModule):
    def __init__(self, encoder, decoder, train_dataset, lr, batch_size):
        super(VGAELightning, self).__init__()
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

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)
    

    def kl_loss(self, z, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        z = self.encode(x, edge_index)
        loss = self.kl_loss(z, self.__mu__, self.__logstd__)
        if __name__ == "__main__":
            wandb.log({"train_loss": loss})
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
        if __name__ == "__main__":
            wandb.log({"test_loss": loss})
        return loss
    
    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
class PositionEncoder(nn.Module):
    def __init__(self, ConvLayer, in_channels, hidden_channels, out_channels=2):
        super(PositionEncoder, self).__init__()
        self.conv_shared = ConvLayer(in_channels, hidden_channels)
        self.conv_mu = ConvLayer(hidden_channels, out_channels)
        self.conv_logvar = ConvLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv_shared(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

class PositionAutoEncoderLightning(pl.LightningModule):
    def __init__(self, encoder, decoder, train_dataset, lr, batch_size):
        super(PositionAutoEncoderLightning, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.lr = lr

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)

    def encode(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        return self.reparametrize(mu, logvar)

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        z = self.encode(x, edge_index)
        decoded = self.decode(z, edge_index)
        loss = F.mse_loss(decoded, x) # Use mean squared error as the loss function
        self.log("train_loss", loss)
        return loss
    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        z = self.encode(x, edge_index)
        decoded = self.decode(z, edge_index)
        loss = F.mse_loss(decoded, x)
        self.log("test_loss", loss)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    
if __name__ == "__main__":
    if True:
        data = MNISTSuperpixels("data/MNISTSuperpixels", pre_transform=pre_transform)
        train_dataset = data[:int(len(data) * 0.8)]
        test_dataset = data[int(len(data) * 0.8):]
        encoder = GenericEncoder(GCNConv, 2, 32, 16)
        decoder = GCNConv(16, 10)
        model = PositionAutoEncoderLightning(encoder, decoder, train_dataset, lr=0.01, batch_size=64)

        trainer = pl.Trainer(max_epochs=10, devices=1)
        trainer.fit(model)
        trainer.test(model)


        torch.save(model.state_dict(), "model.pt")
    else:
        data = MNISTSuperpixels("data/MNISTSuperpixels", pre_transform=pre_transform)

        train_dataset = data[:int(len(data) * 0.8)]
        encoder = GenericEncoder(GCNConv, 2, 32, 16)
        decoder = GCNConv(16, 10)
        model = VGAELightning(encoder, decoder, train_dataset, lr=0.01, batch_size=64)
        model.load_state_dict(torch.load("model.pt"))
        model.eval() # Set model to evaluation mode

        # Sample from the latent space
        z = torch.randn(1, model.encoder.latent_dim) 

        # Use the decoder to generate a graph
        generated_graph = model.decoder(z)

        # Convert the generated graph to networkx for easy visualization
        import networkx as nx
        from torch_geometric.utils.convert import to_networkx
        import matplotlib.pyplot as plt
        nx_graph = to_networkx(generated_graph).to_undirected()

        # Visualize the generated graph
        nx.draw(nx_graph, with_labels=True)
        plt.show()


