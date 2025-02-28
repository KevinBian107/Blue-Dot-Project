import pandas as pd
import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch

import matplotlib.pyplot as plt

from models.ClassicModels import FeedForwardNN, RecurrentNet, LSTMModel
from models.LCModels import LCNECortexFitter, LCNECortexLSTM
from models.LCGadgetModels import FFGadgetController, FFGadgetUncertainController

def plot_loss_curve(loss_history):
        """Plots the training loss curve."""
        plt.figure(figsize=(7,5))
        plt.plot(loss_history, label="Train Loss", color='blue', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid()
        plt.show()

def train_feed_forward_nn(X_train, Y_train, epochs):
    '''Training feed forward neural network'''
    input_size = X_train.shape[1]
    model = FeedForwardNN(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = loss_fn(Y_pred, Y_train.view(-1, 1))
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    print("Training complete!")
    
    return model

def train_vanilla_rnn(X_train, Y_train, epochs):
    '''Training vanilla RNN'''
    # Convert input data into sequences
    
    X_rnn = X_train.unsqueeze(1) 
    Y_rnn = Y_train.unsqueeze(1) 

    print(f"X_rnn Shape: {X_rnn.shape}, Y_rnn Shape: {Y_rnn.shape}")

    model = RecurrentNet(input_size=X_rnn.shape[2])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = model(X_rnn)
        loss = loss_fn(Y_pred, Y_rnn)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("Training complete!")
    
    return model


def train_vanilla_lstm(X_train, Y_train, epochs, hidden_dim):
    '''Training vanilla LSTM'''
    
    input_dim = X_train.shape[1]
    num_layers = 2
    learning_rate = 0.001
    batch_size = 32

    model = LSTMModel(input_dim, hidden_dim)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze(1)  # Convert to (batch_size, seq_length=1, input_dim)

        idx = torch.randint(0, X_train.shape[0], (batch_size,))
        X_batch, Y_batch = X_train[idx], Y_train[idx]

        Pupil_pred, _, _= model(X_batch)

        loss = loss_fn(Pupil_pred, Y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    print("Training complete!")
    return model


def train_vanilla_lc_model(X_train, Y_train, epochs):
    '''Training vanilla LC model'''
    input_dim = X_train.shape[1]  # Dynamically get input feature size
    model = LCNECortexFitter(input_dim=input_dim, hidden_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    batch_size = 32

    for epoch in range(epochs):
        optimizer.zero_grad()

        idx = torch.randint(0, X_train.shape[0], (batch_size,))
        X_batch, Y_batch = X_train[idx], Y_train[idx]

        prev_LC = torch.zeros(batch_size, 8)  
        prev_Cortex = torch.zeros(batch_size, 8)  

        LC_pred, NE_pred, C_pred, Pupil_pred = model(X_batch, prev_LC, prev_Cortex)

        loss = loss_fn(Pupil_pred, Y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    print("Training complete!")
    
    return model

def train_lstm_lc_model(X_train, Y_train, epochs, hidden_dim):
    '''Training LSTM LC model'''
    input_dim = X_train.shape[1]
    model = LCNECortexLSTM(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    batch_size = 32

    for epoch in range(epochs):
        optimizer.zero_grad()

        idx = torch.randint(0, X_train.shape[0], (batch_size,))
        X_batch, Y_batch = X_train[idx], Y_train[idx]

        prev_LC = torch.zeros(batch_size, hidden_dim)
        prev_Cortex = torch.zeros(batch_size, hidden_dim)
        cell_state = torch.zeros(batch_size, hidden_dim)

        LC_pred, NE_pred, C_pred, Pupil_pred, cell_state = model(X_batch, prev_LC, prev_Cortex, cell_state)
        loss = loss_fn(Pupil_pred, Y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()
        
        scheduler.step(loss) 

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    print("Training complete!")
    
    return model


def train_ff_controller(X_train, Y_train, epochs, hidden_dim, patience=2000):
    """Train the FF Controller model with LC-NE gadget"""
    
    input_dim = X_train.shape[1]
    batch_size = min(32, X_train.shape[0])  # Adjust batch size if dataset is small
    model = FFGadgetController(input_dim, hidden_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    loss_fn = nn.SmoothL1Loss()

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        idx = torch.randint(0, X_train.shape[0], (batch_size,))
        X_batch, Y_batch = X_train[idx], Y_train[idx]

        Pupil_pred, LC_t, NE_t, tonic_NE, phasic_NE = model(X_batch)
        loss = loss_fn(Pupil_pred, Y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        scheduler.step(loss)  

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Training complete!")
    
    return model


def train_ff_uncertain_controller(X_train, Y_train, epochs, hidden_dim, patience=200, batch_size=32):
    """Train the FF Controller model with LC-NE gadget for pupil dilation + uncertainty."""
    
    def aleatoric_loss(y_pred, y_true, exp_var):
        """Aleatoric uncertainty loss (adaptive uncertainty modeling)."""
        
        loss = torch.mean(torch.exp(-exp_var) * (y_pred - y_true) ** 2 + exp_var)
        return loss
    
    input_dim = X_train.shape[1]
    batch_size = min(batch_size, X_train.shape[0])
    model = FFGadgetUncertainController(input_dim, hidden_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # loss_fn = nn.SmoothL1Loss()
    # loss_fn = nn.GaussianNLLLoss()
    loss_fn = aleatoric_loss
    
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        idx = torch.randint(0, X_train.shape[0], (batch_size,))
        X_batch = X_train[idx]
        Y_pupil_batch = Y_train[idx].unsqueeze(1)

        pupil_mean, pupil_var = model(X_batch)

        # loss = loss_fn(pupil_mean, Y_pupil_batch)
        loss = loss_fn(pupil_mean, Y_pupil_batch, pupil_var)
        loss_history.append(loss.item()) 

        loss.backward()
        optimizer.step()
        scheduler.step(loss)  

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # if epoch % 100 == 0:
        #     print(f"[Epoch {epoch}] Variance Min: {pupil_var.min().item()}, Max: {pupil_var.max().item()}")
        #     squared_error = ((pupil_mean - Y_pupil_batch) ** 2).mean().item()
        #     log_var_term = torch.log(pupil_var).mean().item()
        #     print(f"Squared Error Term: {squared_error}, Log Variance Term: {log_var_term}")

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Training complete!")

    plot_loss_curve(loss_history)
    
    return model

