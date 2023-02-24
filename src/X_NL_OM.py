# %% [markdown]
# # Code describe the line of reasoning we need to show for all the cases

# %%
# https://anbasile.github.io/posts/2017-06-25-jupyter-venv/

import matplotlib.pyplot as plt
import numpy as np

# %%
import pandas as pd
from scipy.stats import linregress
from sklearn import datasets, linear_model, model_selection
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# # Load data

# %%
B = pd.read_parquet("./data/B_HKML.parquet")

# %%
B = B[B["date"] < "2022-05-01"]

# %%
X_df = pd.read_parquet("./data/X_HKML.parquet")

# %%
X_df = X_df[X_df["date"] < "2022-05-01"]

# %%
y_df = pd.read_parquet("./data/Y_HKML.parquet")

# %%
y_df = y_df.iloc[X_df.index]

# %% [markdown]
# # Train-test split

# %%
moons = X_df.date.unique()

# %%
moon_train = moons[:230]
moon_test = moons[260:]

# %% [markdown]
# # Define loss function

# %%
error_lis = []

# %%
loss_mse = nn.MSELoss()


def custom_loss2(y_pred, y_true, b_moon):
    import pdb as pdb

    pdb.set_trace()
    Omega = torch.eye(len(y_true)) - torch.mm(b_moon, torch.pinverse(b_moon))  #
    z_hat = torch.mm(Omega, y_pred)  # torch.Size([996, 1])

    # error = torch.sum((z_hat - y_true)**2) / len(y_true)
    # error = torch.mm((z_hat - y_true).T, z_hat - y_true)/len(y_true)
    error = torch.linalg.norm(z_hat - y_true) / len(y_true)
    # error_lis.append(error)
    # error = error1 + error2
    # error1 --> mse(y_hat, y_true)
    # error2 --> mse(0, torch.mm(b_moon, torch.pinverse(b_moon)))

    return error


def custom_loss3(y_pred, y_true, b_moon):
    # import pdb as pdb
    # pdb.set_trace()
    # error = error1 + error2
    error1 = loss_mse(y_hat, y_true)
    error2 = loss_mse(0, torch.mm(b_moon.T, y_pred))
    error = error1 + error2

    return error / len(y_true)


# %% [markdown]
# # Train model and predict

import numpy as np
import pandas as pd

# %%
import torch
import torch.nn as nn
from tqdm import tqdm

# %%
# As this notebook shows a simple linear model, there is no need to introduce validation data, i.e.
# the training is deterministic and only one epoch is necessary to perform it.
# Nevertheless, this section should train such that the validation loss is minimized.

# %%
class RegressionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        hidden = self.fc1(x)
        relu_output = self.relu(hidden)
        # dropout = self.dropout(relu_output)
        hidden2 = self.fc2(relu_output)
        relu_output = self.relu(hidden2)
        # dropout = self.dropout(relu_output)
        y_pred = self.fc3(relu_output)
        return y_pred


# %%
# Create an instance of the model
model = RegressionModel(
    input_size=160, hidden_size=512, output_size=1
)  # hidden size ==> 500 ==> loss: inf ==> exploding gradient


# Define the optimizer
optimizer = torch.optim.Adam(model.parameters())

# train model for 5 epochs using custom loss function
num_epochs = 20

# %%
for i in tqdm(range(num_epochs)):
    print(f"Epoch {i+1}")
    # create minibatch of data
    # batch_size = 100
    for moon in moons[:230]:
        # pdb.set_trace()
        X_moon = torch.from_numpy(
            X_df[X_df.date == moon].drop(columns=["date"]).values
        ).to(torch.float32)
        y_moon = torch.from_numpy(y_df[X_df.date == moon].values.reshape(-1, 1)).to(
            torch.float32
        )

        b_moon = torch.from_numpy(
            B[X_df.date == moon].drop(columns=["date"]).values
        ).to(torch.float32)
        # import pdb
        # pdb.set_trace()
        y_pred = model(X_moon)
        loss = custom_loss3(y_pred, y_moon, b_moon)
        # loss = loss_mse(y_pred, y_moon)
        error_lis.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss every 100 epochs
    if (i + 1) % 5 == 0:
        print(f"Epoch [{i+1}/{num_epochs}], Loss: {loss.item():.4f}")

plt.plot(error_lis)

# plt.plot(error_lis)

y_hat = (
    model(
        torch.from_numpy(X_df[X_df.date > moons[260]].drop(columns=["date"]).values).to(
            torch.float32
        )
    )
    .detach()
    .numpy()
)
y_test = y_df[X_df.date > moons[260]].values.reshape(-1, 1)


slope, intercept, r_value, p_value, std_err = linregress(y_test, y_hat)
plt.scatter(y_test, y_hat)
plt.plot(y_test, slope * y_test + intercept, "r")
plt.show()

y_hat_df = pd.DataFrame()
y_hat_df["date"] = X_df[X_df["date"] > moons[260]].date
y_hat_df["y_hat"] = y_hat
y_hat_df["Y"] = y_test

y_hat_df  # custom_fitenss

y_hat_df  # mse_fitness


def spear(x):
    return x.corr(method="spearman").iloc[0, 1]


spearman = y_hat_df.groupby("date").apply(lambda x: spear(x))

spearman.mean(), spearman.shape  # 46 values for 46 moons

spearman.plot()
plt.axhline(y=spearman.mean(), color="r", linestyle="-")
plt.show()

spearman.hist(bins=30)
plt.axvline(x=spearman.mean(), color="r", linestyle="-")

mse = y_hat_df.groupby("date").apply(lambda x: mean_squared_error(x.y_hat, x.Y))
mse.plot()
plt.axhline(y=mse.mean(), color="r", linestyle="-")
plt.show()

mse.hist(bins=30)
plt.axvline(x=mse.mean(), color="r", linestyle="-")

B = B[X_df.date > moons[260]]

y_hat_date = y_hat_df[["date", "y_hat"]]


# %%
import sys

sys.path.insert(1, "/Users/utkarshpratiush/Cr_D/Feature engg/feature-engineering/src")
from class_ import Data

# %%
data = Data(f_matrix=y_hat_date, b_matrix=B)

# %%
data.orthogonalize()

# %%
data.f_matrix

# %%
y_hat_df.y_hat = data.f_matrix.y_hat

# %%
y_hat_df

# %%
slope, intercept, r_value, p_value, std_err = linregress(y_hat_df.Y, y_hat_df.y_hat)
plt.scatter(y_hat_df.Y, y_hat_df.y_hat)
plt.plot(y_hat_df.Y, slope * y_hat_df.Y + intercept, "r")
plt.show()

# %%
spearman = y_hat_df.groupby("date").apply(lambda x: spear(x))
spearman.plot()
plt.axhline(y=spearman.mean(), color="r", linestyle="-")
plt.show()

# %%
spearman.hist(bins=30)
plt.axvline(x=spearman.mean(), color="r", linestyle="-")

# %%
mse = y_hat_df.groupby("date").apply(lambda x: mean_squared_error(x.y_hat, x.Y))
mse.plot()
plt.axhline(y=mse.mean(), color="r", linestyle="-")
plt.show()

# %%
mse.hist(bins=30)
plt.axvline(x=mse.mean(), color="r", linestyle="-")

# %%
