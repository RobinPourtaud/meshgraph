import torch
import clip 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb

# https://github.com/openai/CLIP
# And thanks to https://github.com/openai/CLIP/issues/83


def train(data, config_model):
  wandb.init(project="autoencoder", config=config_model)

  BATCH_SIZE = 32
  EPOCH = 30
  DEVICE = config_model["device_clip"]
  model, _ = clip.load("RN50", device=DEVICE, jit=False)

  print(data)
  train_dataloader = DataLoader(data, batch_size = BATCH_SIZE) #Define your own dataloader

  def convert_models_to_fp32(model): 
      for p in model.parameters(): 
          p.data = p.data.float() 
          p.grad.data = p.grad.data.float() 


  if DEVICE == "cpu":
    model.float()
  else :
    clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

  loss_imgf = nn.CrossEntropyLoss()
  loss_txtf = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

  # add your own code to track the training progress.
  for epoch in range(EPOCH):
    wandb.log({"epoch": epoch})
    for batch in train_dataloader :
        optimizer.zero_grad()

        images,texts = batch 
      
        images= images.to(DEVICE)
        texts = texts.to(DEVICE)
      
        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=DEVICE)
        loss_img = loss_imgf(logits_per_image,ground_truth)
        loss_txt = loss_txtf(logits_per_text,ground_truth)
        

        total_loss = (loss_img + loss_txt)/2
        wandb.log({"loss_per_batch": total_loss})
        wandb.log({"loss_graph_per_batch": loss_img})
        wandb.log({"loss_txt_per_batch": loss_txt})
        total_loss.backward()
        if DEVICE == "cpu":
          optimizer.step()
        else : 
          convert_models_to_fp32(model)
          optimizer.step()
          clip.model.convert_weights(model)
    wandb.log({"loss": total_loss})
    wandb.log({"loss_graph": loss_img})
    wandb.log({"loss_txt": loss_txt})
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"model_checkpoint/model_{epoch}.pt")



def generate(data, config_model):
  from torchvision.datasets import mnist
  wandb.init(project="autoencoder", config=config_model)
  device = config_model["device_clip"]

  for model_n in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
    model, preprocess = clip.load(model_n, device=device, jit=False)
    dataMnist = mnist.MNIST(root="data", download=True, transform=preprocess)

    topk_accs = {k: 0 for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    for i in range(len(dataMnist)):
      model.eval()
      model.to(device)

      # Prepare the inputs
      image, class_id = dataMnist[i]

      image_input = image.to(device).unsqueeze(0)
      classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
      text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
      # Calculate features
      with torch.no_grad():
          image_features = model.encode_image(image_input)
          text_features = model.encode_text(text_inputs)

      
      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
      values, indices = similarity[0].topk(10)

      
      

    #table = wandb.Table(columns=["topk", "accuracy"])
    #for k, v in topk_accs.items():
    #  table.add_data(k, str(v / len(dataMnist)))
    #wandb.log({"topk_accs_" + model_n: table})
      
      
