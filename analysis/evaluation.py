import pandas as pd
import numpy as np
import torch

from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

from models.ClassicModels import FeedForwardNN, RecurrentNet, LSTMModel
from models.LCModels import LCNECortexFitter, LCNECortexLSTM
from models.LCGadgetModels import FFGadgetController, FFGadgetUncertainController

from scipy.stats import pearsonr


def evaluate_model(model, X_test, Y_test, df_clean, scaler_Y=None):
    """
    Evaluates different types of models (FeedForwardNN, RNN, LSTM, LCNECortex).
    Automatically adjusts for model-specific structures.

    Args:
        model: Trained model
        X_test: Input tensor
        Y_test: Ground truth tensor
        df_clean: Original DataFrame with 'Event_PupilDilation' column
        scaler_Y: Optional scaler for inverse transformation

    Returns:
        None (Generates plots & prints Pearson correlation)
    """
    model.eval()
    batch_size = X_test.shape[0]

    # ---- MODEL TYPE DETECTION ---- #
    model_name = type(model).__name__
    print(f"Evaluating Model: {model_name}")

    # ---- FEEDFORWARD NN ---- #
    if isinstance(model, FeedForwardNN):
        with torch.no_grad():
            Y_pred = model(X_test).cpu().numpy()

        Y_test_actual = Y_test.cpu().numpy().reshape(-1, 1)
        if scaler_Y:
            Y_pred = scaler_Y.inverse_transform(Y_pred)
            Y_test_actual = scaler_Y.inverse_transform(Y_test_actual)

    # ---- RECURRENT NN ---- #
    elif isinstance(model, (RecurrentNet, LSTMModel)):
        X_rnn = X_test.unsqueeze(1) 
        
        with torch.no_grad():
            if isinstance(model, RecurrentNet):
                Y_pred = model(X_rnn).cpu().numpy()
                
            elif isinstance(model, LSTMModel):
                if len(X_test.shape) == 2:
                    X_test = X_test.unsqueeze(1)  # Ensure it has (batch_size, seq_length, input_dim)
                Y_pred, _, _ = model(X_test)

        if scaler_Y:
            Y_pred = scaler_Y.inverse_transform(Y_pred).reshape(-1, 1)

    # ---- LCNECortex Variants ---- #
    elif isinstance(model, (LCNECortexLSTM, LCNECortexFitter, FFGadgetController, FFGadgetUncertainController)):
        prev_LC = torch.zeros(batch_size, model.hidden_dim)
        prev_Cortex = torch.zeros(batch_size, model.hidden_dim)
        cell_state = torch.zeros(batch_size, model.hidden_dim)

        with torch.no_grad():
            
            if isinstance(model, LCNECortexLSTM):
                _, _, _, Pupil_pred, _ = model(X_test, prev_LC, prev_Cortex, cell_state)
            
            elif isinstance(model, FFGadgetController):
                Pupil_pred, _, _, _, _ = model(X_test)
            
            elif isinstance(model, FFGadgetUncertainController):
                Pupil_pred, _ = model(X_test)
            
            elif isinstance(model, LCNECortexFitter):
                _, _, _, Pupil_pred = model(X_test, prev_LC, prev_Cortex)
                

        Y_pred = Pupil_pred.cpu().numpy().reshape(-1, 1)
        if scaler_Y:
            Y_pred = scaler_Y.inverse_transform(Y_pred)

    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    # ---- CREATE DATAFRAMES ---- #
    df_predictions = pd.DataFrame({'PupilPred': Y_pred.flatten()})
    df_actual = df_clean[['Event_PupilDilation']].reset_index(drop=True)

    min_length = min(len(df_actual), len(df_predictions))
    df_predictions = df_predictions.iloc[:min_length]
    df_actual = df_actual.iloc[:min_length]

    # ---- SCATTER PLOT: Actual vs. Predicted ---- #
    plt.figure(figsize=(6, 4))
    plt.scatter(df_actual['Event_PupilDilation'], df_predictions['PupilPred'], alpha=0.6, edgecolors="k")
    plt.xlabel("Actual Pupil Dilation")
    plt.ylabel("Predicted Pupil Dilation")
    plt.title(f"{model_name}: Actual vs. Predicted Pupil Dilation")
    plt.grid(True)
    plt.show()

    # ---- HISTOGRAM: Distribution of Pupil Dilation ---- #
    plt.figure(figsize=(8, 4))
    sns.histplot(df_actual['Event_PupilDilation'], kde=True, label="Actual", color='blue', alpha=0.6, bins=50)
    sns.histplot(df_predictions['PupilPred'], kde=True, label="Predicted", color='red', alpha=0.6, bins=50)
    plt.xlabel("Pupil Dilation")
    plt.ylabel("Count")
    plt.legend()
    plt.title(f"{model_name}: Distribution of Actual vs. Predicted Pupil Dilation")
    plt.grid(True)
    plt.show()

    # ---- COMPUTE PEARSON CORRELATION ---- #
    corr, _ = pearsonr(df_actual['Event_PupilDilation'], df_predictions['PupilPred'])
    print(f"{model_name} - Pearson Correlation: {corr:.4f}")
