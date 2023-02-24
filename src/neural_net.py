import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm


X = pd.read_parquet("../data/X_HKML.parquet") # [345147 rows x 161 columns]
B = pd.read_parquet("../data/B_HKML.parquet") # [345147 rows x 86 columns]
XB = pd.read_parquet("../data/XB_PCA_HKML.parquet") # [310704 rows x 178 columns]
Y = pd.read_parquet("../data/Y_HKML.parquet") # [345147 rows x 1 columns]

#X = XB
moons = X.date.unique()

class RegressionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        hidden = self.fc1(x)
        relu_output = self.relu(hidden)
        dropout = self.dropout(relu_output)
        hidden2 = self.fc2(dropout)
        relu_output = self.relu(hidden2)
        dropout = self.dropout(relu_output)
        y_pred = self.fc3(dropout)
        return y_pred

error_lis = []

def custom_loss2(y_pred, y_true, b_moon):
    Omega = torch.eye(len(y_true)) - torch.mm(b_moon, torch.pinverse(b_moon))
    z_hat = torch.mm(Omega, y_pred)
    
    #error = torch.sum((z_hat - y_true)**2) / len(y_true)
    #error = torch.mm((z_hat - y_true).T, z_hat - y_true)/len(y_true)
    error = torch.linalg.norm(z_hat - y_true)
    error_lis.append(error)
    
    return error

# Create an instance of the model
model = RegressionModel(input_size=160, hidden_size=512, output_size=1) # hidden size ==> 500 ==> loss: inf ==> exploding gradient

loss_mse = nn.MSELoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# train model for 5 epochs using custom loss function
num_epochs = 10
for i in tqdm(range(num_epochs)):
    print(f"Epoch {i+1}")
    # create minibatch of data
    #batch_size = 100
    for moon in tqdm(moons):
        #pdb.set_trace()
        X_moon = torch.from_numpy(X[X.date == moon].drop(columns=["date"]).values).to(torch.float32)
        y_moon = torch.from_numpy(Y[X.date == moon].values.reshape(-1,1)).to(torch.float32)

        b_moon = torch.from_numpy(B[B.date == moon].drop(columns=["date"]).values).to(torch.float32)
        #import pdb
        #pdb.set_trace()
        y_pred = model(X_moon)
        loss = custom_loss2(y_pred, y_moon, b_moon)
        #loss = loss_mse(y_pred, y_moon)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Print the loss every 100 epochs
    if (i + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1, num_epochs, loss.item()))

pred_train = model(torch.from_numpy(X.drop(columns=["date"]).values).to(torch.float32))
print(loss_mse(pred_train, torch.from_numpy(Y.values[:, -1].reshape(-1,1)).to(torch.float32)).item())
#print(custom_loss2(pred_train, torch.from_numpy(Y.values[:, -1].reshape(-1,1)).to(torch.float32)).item())

# X, Y : custom loss trained: mse-> 0.014834950678050518
# X, Y : mse trained: mse-> 0.001293371431529522 
# XB, Y 

import pdb
pdb.set_trace()