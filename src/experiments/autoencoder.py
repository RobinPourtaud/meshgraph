import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.models.autoencoder import VGAE, InnerProductDecoder, ARGA, ARGVA
from torch_geometric.loader import DataLoader
from torch_geometric import transforms as T
import networkx as nx
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning.loggers import WandbLogger

class LightingAutoEncoder(pl.LightningModule):
    def __init__(self, config):
        super(LightingAutoEncoder, self).__init__()
        
        # Encoder
        encoder_layer_name : str = config.encoder_layer_name
        self.encoder_in_channels = config.encoder_input_size
        self.encoder_hidden_channels = config.encoder_hidden_size
        self.encoder_out_channels = config.encoder_output_size
        encoder_layer_names = {
            "GCNConv": GCNConv,
            "GATConv": GATConv
        }
        self.encoder_layer = encoder_layer_names[encoder_layer_name]
        self.encoder = GenericEncoder(self.encoder_layer, self.encoder_in_channels, self.encoder_hidden_channels, self.encoder_out_channels)
            
        # Decoder
        self.decoder = InnerProductDecoder()

        # Model
        self.model = VGAE(self.encoder, self.decoder)

        # Hyperparameters
        self.lr = config.lr
        self.optimizer = config.optimizer

    def get_embeddings(self, x, edge_index):
        return self.model.encode(x, edge_index)
    
    def forward(self, x, edge_index):
        z = self.model.encode(x, edge_index) # encode the graph, and reparameterize
        return z

    def training_step(self, batch, batch_idx):
        z = self.forward(batch.x, batch.edge_index)
        loss = self.model.recon_loss(z, batch.edge_index) + (1 / batch.num_nodes) * self.model.kl_loss()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer not supported")
     
class GenericEncoder(nn.Module):
    def __init__(self, encoder_layer, in_channels, hidden_channels, out_channels):
        super(GenericEncoder, self).__init__()
        self.conv_shared = encoder_layer(in_channels, hidden_channels)
        self.conv_mu = encoder_layer(hidden_channels, out_channels)
        self.conv_logvar = encoder_layer(hidden_channels, out_channels)
        self.latent_dim = out_channels

    def forward(self, x, edge_index):
        x = F.relu(self.conv_shared(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar
    

def train(data, config_model):
    """Train the autoencoder using wandb sweeps and pytorch lightning.
    The config file is located in configs/autoencoder/train.yaml
    
    Args:
        data (torch_geometric.data.Data): The dataset to train on.
        config_data (dict): The config file for the dataset.
        config_model (dict): The config file for the model.

    Returns:
        None
    """
    def train_sweep():
        wandb_logger = WandbLogger(project="autoencoder", config=config_model)
        # print all attributes of config
        
        data_loader = DataLoader(data, batch_size=wandb.config.batch_size, num_workers=wandb.config.num_workers)

        model = LightingAutoEncoder(wandb.config)
        trainer = pl.Trainer(max_epochs=wandb.config.epochs, logger=wandb_logger, devices=wandb.config.devices)
        trainer.fit(model, data_loader)
        # save the whole model in model_path config folder
        try:
            torch.save(model, wandb.config.model_path + f"/{wandb.run.name}.pt")
        except:
            print("Could not save model")
        


    sweep_id = wandb.sweep(config_model, project="autoencoder")
    wandb.agent(sweep_id, train_sweep)
    

def generate(data, config):
    """Generate a graph using the trained autoencoder.
    
    Args:
        data (torch_geometric.data.Data): The dataset to train on.
        config (dict): The config file for the model.

    Returns:
        torch_geometric.data.Data: The generated graph.
    """
    from torch_geometric.utils.convert import to_networkx

    wandb.init(project="autoencoder", config=config)
    print("Generating graph...")
    # load mnist original dataset just to visualize the difference betzeen the images and the generated images
    from torchvision import datasets
    mnist = datasets.MNIST('data', train=True, download=True)
    import numpy as np
    for model_path in config["model_paths"]:
        model = torch.load(model_path)
        model.eval()
        # print the input graph and output graph, and maybe the reconstruction loss as title, config["generateNumber"] times
        # random sample from the dataset
        for graphId in range(config["generateNumber"]):
            graph = data[graphId]
            z = model(graph.x, graph.edge_index)
            probabilities_of_edges = InnerProductDecoder().forward_all(z, sigmoid=True)

            # Convert probabilities to binary (0 or 1) using threshold 0.5
            binary_adjacency_matrix = (probabilities_of_edges > config["probability_of_edge"]).float()
            # set diagonal to 0
            binary_adjacency_matrix = binary_adjacency_matrix - torch.diag(torch.diag(binary_adjacency_matrix))
            print(binary_adjacency_matrix)
            
            # plot and send to wandb
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
            fig.suptitle(f"Input graph and generated graph for the number {graph.y.item()} graph")
            GOri = to_networkx(graph, to_undirected=True)

            posOri = {
                i : graph.x[i].numpy() for i in range(len(graph.x))
            }
            # plot the image on the first axis
            img = np.array(mnist[graphId][0])
            ax0.imshow(img, cmap="gray")
            nx.draw(GOri, pos=posOri, ax=ax1)
            nx.draw(nx.from_numpy_array(binary_adjacency_matrix.cpu().detach().numpy()), ax=ax2, pos=posOri)
            # log image to wandb
            wandb.log({"generated_graph": wandb.Image(fig)})
            plt.close(fig)
    print("Done generating graphs")


    
