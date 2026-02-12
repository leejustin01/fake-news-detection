import numpy as np
from models.nn import FeedForwardNetwork
from models.logreg import LogisticRegression
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import datetime
import random
import string
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "bs":32,   # batch size
    "lr":0.003, # learning rate
    "l2reg":0.00005, # weight decay
    "lr_decay":0.95, # exponential learning decay
    "max_epoch":5,
    "embedding_dim":384, # dimension of SBERT embeddings
    "layer_widths":[32, 32], # widths of hidden layers for FeedForwardNetwork
    "output_dim":1
}

def generateRunName():
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    now = datetime.datetime.now()
    run_name = "["+random_string+"] CS406  "+now.strftime("[%m-%d-%Y--%H:%M]")
    return run_name


def train(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_decay"])
    loss_fn = nn.BCELoss()

    pbar = tqdm(total=config["max_epoch"]*len(dataloader), desc="Training Iterations", unit="batch")
    for epoch in range(config["max_epoch"]):
        for x, y in dataloader:
            x.to(device)
            y.to(device)
            
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = loss_fn(y_pred.view(-1), y)
            loss.backward()
            optimizer.step()
            pbar.update(1)
        scheduler.step()
        
        
        
if __name__ == "__main__":
    embeddings = np.load("./data/processed/X_train.npy")
    labels = np.load("./data/processed/y_train.npy")
    
    X = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels).float()
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=config["bs"], shuffle=True)

    ffnn = FeedForwardNetwork(config["embedding_dim"], config["layer_widths"], config["output_dim"])
    logreg = LogisticRegression(config["embedding_dim"], config["output_dim"])
    ffnn.to(device)
    logreg.to(device)
    
    print("Training Neural Network")
    train(ffnn, dataloader)
    print("Training Logistic Regression")
    train(logreg, dataloader)
    
    torch.save(ffnn.state_dict (), "./models/chkpts/nn")
    torch.save(logreg.state_dict(), "./models/chkpts/logreg")




