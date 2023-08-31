import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader


def get_perceptron_weights(X_train, y_train, num_epochs=20):
    n_dims = X_train.shape[1]

    X_train = torch.tensor(X_train.values).float()
    y_train = torch.tensor(y_train).long()


    train_set = TensorDataset(X_train, y_train)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    model = nn.Sequential(nn.Linear(n_dims, 2), nn.Softmax(dim=1))
    # define the loss function
    loss_fn = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # put the model in training mode
    model.train()

    for epoch in range(num_epochs):
        for X_train, Y_train in train_loader:
                # compute the model output
                Y_pred = model(X_train)
                # calculate loss
                loss = loss_fn(Y_pred, Y_train)
                # reset the gradients
                optimizer.zero_grad()
                # backpropagation
                loss.backward()
                # update model weights
                optimizer.step()

    return abs(model[0].weight.data[1].numpy())
