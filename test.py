import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.nn import FeedForwardNetwork
from train import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def computeAccuracy(out, y):
  _, predicted = torch.max(out.data, 1)
  acc = (predicted == y).sum().item()/out.shape[0]
  return acc

def evaluate(model, dataloader):
    model.eval()

    running_acc = 0

    with torch.no_grad():
        for x,y in dataloader:

            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)

            out = model(x)
            
            preds = (out > 0.5).float()
            acc = (preds == y).sum().item()

            running_acc += acc
    return running_acc/len(dataloader.dataset)

if __name__ == "__main__":
    model = FeedForwardNetwork(config["embedding_dim"], config["layer_widths"], config["output_dim"])
    checkpoint = torch.load("./models/chkpts/ffnn", map_location=device)
    model.load_state_dict(checkpoint)
    
    embeddings = np.load("./data/processed/X_train.npy")
    labels = np.load("./data/processed/y_train.npy")
    
    X = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels).float()
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config["bs"], shuffle=True)

    test_acc = evaluate(model, dataloader)

    print(f"Test Accuracy: {test_acc}")