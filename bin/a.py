from experiment.datasetShapenet import ShapeNet
import torch
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.multiprocessing as mp


class GATAutoencoder(pl.LightningModule):
    def __init__(self, num_features, hidden_dim, embed_dim):
        super().__init__()
        self.encoder = GATConv(num_features, hidden_dim)
        self.embedder = GATConv(hidden_dim, embed_dim)
        self.decoder = GATConv(embed_dim, num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.encoder(x, edge_index))
        x = F.relu(self.embedder(x, edge_index))
        x = self.decoder(x, edge_index)
        return x

    def training_step(self, batch, batch_idx):
        x_hat = self(batch)
        loss = F.mse_loss(x_hat, batch.x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Instantiate the model
model = GATAutoencoder(num_features=8, hidden_dim=16, embed_dim=8)

# Load the dataset
dataset = ShapeNet(root='data/processed/pytorch_graph', transform=None, pre_transform=None)
dataset = dataset.shuffle()
train_dataset = dataset[:20]  # Use only 50 graphs for training

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=4)

# Define the trainer and train the model
trainer = pl.Trainer(max_epochs=10, devices=[1])
trainer.fit(model, train_loader)
