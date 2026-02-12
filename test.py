import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.nn import FeedForwardNetwork
from models.logreg import LogisticRegression
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
    ffnn = FeedForwardNetwork(config["embedding_dim"], config["layer_widths"], config["output_dim"])
    logreg = LogisticRegression(config["embedding_dim"], config["output_dim"])
    
    ffnn_checkpoint = torch.load("./models/chkpts/nn", map_location=device)
    logreg_checkpoint = torch.load("./models/chkpts/logreg", map_location=device)
    ffnn.load_state_dict(ffnn_checkpoint)
    logreg.load_state_dict(logreg_checkpoint)
    
    embeddings = np.load("./data/processed/X_train.npy")
    labels = np.load("./data/processed/y_train.npy")
    
    X = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels).float()
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config["bs"], shuffle=True)

    ffnn_test_acc = evaluate(ffnn, dataloader)
    logreg_test_acc = evaluate(logreg, dataloader)

    print(f"Neural Network Test Accuracy: {ffnn_test_acc}")
    print(f"Logistic Regression Test Accuracy: {logreg_test_acc}")
    print(f"Diff: {ffnn_test_acc - logreg_test_acc}")