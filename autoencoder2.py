import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric import transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models.autoencoder import VGAE, InnerProductDecoder, ARGA, ARGVA
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F

TRAIN = False
PREDICT = not TRAIN and True
LR = 0.001
EPOCHS = 30
BATCH_SIZE = 64
IN_CHANNELS = 2
HIDDEN_CHANNELS = 16 # Or EMBEDDING_DIM
OUT_CHANNELS_ENCODER = 16
ENCODER = GCNConv
DECODER = InnerProductDecoder
AUTOENCODER = VGAE
DATASET = MNISTSuperpixels
DATASET_PATH = "data/MNISTSuperpixels"
LOG = True
DEVICES = 3
NUM_WORKERS = 1
WANDB_PROJECT = "autoencoder"
SAVE_MODEL = True
MODEL_PATH = "models/autoencoder.pt"



def pre_transform(data):
    edge_index = data.edge_index
    edges_mask = (data.x[edge_index[0]] >= 0.65) & (data.x[edge_index[1]] >= 0.65)
    edges_to_keep = torch.nonzero(edges_mask, as_tuple=True)[0]
    data.edge_index = edge_index[:, edges_to_keep]
    data.x = data.pos
    data = T.RemoveIsolatedNodes()(data)
    return data

class LightingAutoEncoder(pl.LightningModule):
    def __init__(self, encoder_layer, decoder_layer, in_channels, hidden_channels, out_channels, lr):
        super(LightingAutoEncoder, self).__init__()
        self.encoder = encoder_layer(in_channels, hidden_channels, out_channels)
        self.model = VGAE(self.encoder, decoder_layer())
        self.lr = lr

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.model.encode(x, edge_index)
        return z

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        z = self.forward(batch)
        loss = self.model.recon_loss(z, edge_index) + (1 / batch.num_nodes) * self.model.kl_loss()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        transform = T.Compose([pre_transform, T.Cartesian()])
        dataset = DATASET(DATASET_PATH, transform=transform)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    def test_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        z = self.forward(batch)
        loss = self.model.recon_loss(z, edge_index) + (1 / batch.num_nodes) * self.model.kl_loss()
        self.log("test_loss", loss)
        return loss

    def test_dataloader(self):
        transform = T.Compose([pre_transform, T.Cartesian()])
        dataset = DATASET(DATASET_PATH, transform=transform)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            
class GenericEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GenericEncoder, self).__init__()
        self.conv_shared = ENCODER(in_channels, hidden_channels)
        self.conv_mu = ENCODER(hidden_channels, out_channels)
        self.conv_logvar = ENCODER(hidden_channels, out_channels)
        self.latent_dim = out_channels

    def forward(self, x, edge_index):
        x = F.relu(self.conv_shared(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

if __name__ == "__main__":
    if LOG:
        import wandb
        wandb_logger = WandbLogger(project=WANDB_PROJECT)

    model = LightingAutoEncoder(GenericEncoder, DECODER, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS_ENCODER, LR)
    if TRAIN:
        trainer = pl.Trainer(devices=DEVICES, max_epochs=EPOCHS, logger=wandb_logger)
        trainer.fit(model)
        trainer.test(model)
        if SAVE_MODEL:
            torch.save(model.state_dict(), MODEL_PATH)

    if PREDICT:
        """ Take 10 random graph from the test set and predict their latent space representation and then decode them back to graph form
            Plot the original graph and the decoded graph with the title the label y
        """
        transform = T.Compose([pre_transform, T.Cartesian()])
        dataset = DATASET(DATASET_PATH, transform=transform)
        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        for batch in loader:
            x, edge_index, y = batch.x, batch.edge_index, batch.y
            z = model(batch)
            edge_index = edge_index.cpu().numpy()
            z = z.cpu().detach().numpy()
            for i in range(len(z)):
                plt.subplot(1, 2, 1)
                G = nx.Graph()
                G.add_nodes_from(range(len(x[i])))
                G.add_edges_from(edge_index.T)
                nx.draw(G, pos=x[i].cpu().numpy(), node_color=y[i].cpu().numpy())
                plt.title("Original Graph")
                plt.subplot(1, 2, 2)
                G = nx.Graph()
                G.add_nodes_from(range(len(z[i])))
                G.add_edges_from(edge_index.T)
                nx.draw(G, pos=z[i], node_color=y[i].cpu().numpy())
                plt.title("Decoded Graph")
                plt.show()
                if LOG:
                    wandb.log({"Original Graph": plt})
                    wandb.log({"Decoded Graph": plt})