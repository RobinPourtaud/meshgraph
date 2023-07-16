
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torchvision import transforms

def generate(data, config):

    TEXTINPUT = config["textInput"]  
    DEVICE = config["device"] 
    import clip 
    import torch
    import torch.optim as optim

    model, preprocess = clip.load("RN50",device=DEVICE,jit=False) #Must set jit=False for training
    checkpoint = torch.load("model_checkpoint/model_0.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    decoder = InnerProductDecoder().forward_all  # decoder model

    # encode the text description using CLIP
    with torch.no_grad():
        text_features = clip.tokenize(TEXTINPUT).to(DEVICE)

    z = torch.randn(16,16).to(DEVICE).requires_grad_()
    
    im = transforms.ToPILImage()(z).convert("RGB")
    im = preprocess(im).to(DEVICE).unsqueeze(0)
   
    optimizer = optim.Adam([z], lr=0.01)

    for step in range(1000):  
        optimizer.zero_grad()
        
        image_features = model.encode_image(im) 
        loss = - torch.cosine_similarity(text_features, image_features).mean()
        
        # perform a gradient update
        loss.backward()
        optimizer.step()
        print(loss)
        if step % 100 == 0:
            print(f'Step {step}, Loss {loss.item()}')

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import wandb
    from torch_geometric.utils import to_networkx
    
    graph = decoder(im.view(16, 16))
    probabilities_of_edges = torch.sigmoid(graph)
    binary_adjacency_matrix = (probabilities_of_edges > config["probabilities_of_edge"]).float()
    binary_adjacency_matrix = binary_adjacency_matrix - torch.diag(torch.diag(binary_adjacency_matrix))


    G = nx.from_numpy_array(binary_adjacency_matrix.cpu().detach().numpy())

    # Set the edges of the NetworkX graph to our binary adjacency matrix
    fig = plt.figure(figsize=(5, 5))
    nx.draw(G, pos=nx.spring_layout(G), node_color="#00b4d9", edge_color="#d9d9d9", width=0.5, with_labels=True)
    wandb.init(project="autoencoder", config=config)
    wandb.log({"generated_graph": wandb.Image(fig)})



