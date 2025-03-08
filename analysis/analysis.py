import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import umap

import matplotlib.pyplot as plt
import seaborn as sns
from ripser import ripser
from persim import wasserstein
from persim import plot_diagrams
import matplotlib.pyplot as plt
import networkx as nx

from scipy.stats import permutation_test
from scipy.stats import ks_2samp, mannwhitneyu
from scipy.stats import pearsonr
import scipy.stats as stats
from scipy.stats import entropy

from models.LCGadgetModels import FFGadgetController, FFGadgetUncertainController
from statsmodels.tsa.stattools import acf, pacf


# ===========================================================================
# Basic dimensionality reduction plots for different models
# ===========================================================================

def perform_pca_and_plot(activations, title, hue_labels):
    '''Do batch PCA plotting, helper graphing functions'''
    
    pca = PCA(n_components=2)
    act_pca = pca.fit_transform(activations)
    explained_variance = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=hue_labels, palette="viridis", alpha=0.7)
    plt.title(f"{title}\nExplained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")
    plt.xlabel(f"PCA Component 1 ({explained_variance[0]:.2f}% Variance)")
    plt.ylabel(f"PCA Component 2 ({explained_variance[1]:.2f}% Variance)")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def pca_feed_forward(model, X_tensor, df_behavior):
    """
    Extract feed forward network's activations, then apply PCA and t-SNE and visualize.
    """
    with torch.no_grad():
        predictions, act1, act2 = model(X_tensor, return_activations=True)

    act1_np = act1.cpu().numpy()
    act2_np = act2.cpu().numpy()
    df_filtered = df_behavior.iloc[:X_tensor.shape[0]].copy()

    pca1 = PCA(n_components=2)
    act1_pca = pca1.fit_transform(act1_np)
    explained_variance1 = pca1.explained_variance_ratio_ * 100

    pca2 = PCA(n_components=2)
    act2_pca = pca2.fit_transform(act2_np)
    explained_variance2 = pca2.explained_variance_ratio_ * 100

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    act1_tsne = tsne.fit_transform(act1_np)
    act2_tsne = tsne.fit_transform(act2_np)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.scatterplot(x=act1_pca[:, 0], y=act1_pca[:, 1], hue=df_filtered["Condition"],
                    palette="viridis", alpha=0.7, ax=axes[0, 0])
    axes[0, 0].set_title(f"PCA Projection of Layer 1 Activations\nExplained Variance: PC1={explained_variance1[0]:.2f}%, PC2={explained_variance1[1]:.2f}%")
    axes[0, 0].set_xlabel(f"PCA Component 1 ({explained_variance1[0]:.2f}% Variance)")
    axes[0, 0].set_ylabel(f"PCA Component 2 ({explained_variance1[1]:.2f}% Variance)")

    sns.scatterplot(x=act2_pca[:, 0], y=act2_pca[:, 1], hue=df_filtered["Condition"],
                    palette="viridis", alpha=0.7, ax=axes[0, 1])
    axes[0, 1].set_title(f"PCA Projection of Layer 2 Activations\nExplained Variance: PC1={explained_variance2[0]:.2f}%, PC2={explained_variance2[1]:.2f}%")
    axes[0, 1].set_xlabel(f"PCA Component 1 ({explained_variance2[0]:.2f}% Variance)")
    axes[0, 1].set_ylabel(f"PCA Component 2 ({explained_variance2[1]:.2f}% Variance)")

    sns.scatterplot(x=act1_tsne[:, 0], y=act1_tsne[:, 1], hue=df_filtered["Condition"],
                    palette="viridis", alpha=0.7, ax=axes[1, 0])
    axes[1, 0].set_title("t-SNE Projection of Layer 1 Activations")
    axes[1, 0].set_xlabel("t-SNE Component 1")
    axes[1, 0].set_ylabel("t-SNE Component 2")

    sns.scatterplot(x=act2_tsne[:, 0], y=act2_tsne[:, 1], hue=df_filtered["Condition"],
                    palette="viridis", alpha=0.7, ax=axes[1, 1])
    axes[1, 1].set_title("t-SNE Projection of Layer 2 Activations")
    axes[1, 1].set_xlabel("t-SNE Component 1")
    axes[1, 1].set_ylabel("t-SNE Component 2")

    plt.tight_layout()
    plt.show()


def pca_lstm(model, X_test, df_clean):
    '''PCA and t-SNE analysis for LSTM model hidden & cell states'''

    # Ensure input shape is (batch_size, seq_length, input_dim)
    if len(X_test.shape) == 2:
        X_test = X_test.unsqueeze(1)

    model.eval()
    with torch.no_grad():
        _, hidden_states, cell_states = model(X_test)

    hidden_states_np = hidden_states.cpu().numpy()
    cell_states_np = cell_states.cpu().numpy()

    scaler = StandardScaler()
    hidden_states_np = scaler.fit_transform(hidden_states_np)
    cell_states_np = scaler.fit_transform(cell_states_np)

    pca_hidden = PCA(n_components=2)
    hidden_pca = pca_hidden.fit_transform(hidden_states_np)
    explained_variance_hidden = pca_hidden.explained_variance_ratio_ * 100

    pca_cell = PCA(n_components=2)
    cell_pca = pca_cell.fit_transform(cell_states_np)
    explained_variance_cell = pca_cell.explained_variance_ratio_ * 100
    
    min_length = min(hidden_pca.shape[0], cell_pca.shape[0], len(df_clean["Condition"]))
    hidden_pca = hidden_pca[:min_length]
    cell_pca = cell_pca[:min_length]
    df_clean = df_clean.iloc[:min_length]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    hidden_tsne = tsne.fit_transform(hidden_pca)
    cell_tsne = tsne.fit_transform(cell_pca)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x=hidden_pca[:, 0], y=hidden_pca[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=axes[0])
    axes[0].set_title(f"LSTM Hidden State PCA\nPC1={explained_variance_hidden[0]:.2f}%, PC2={explained_variance_hidden[1]:.2f}%")
    axes[0].set_xlabel(f"PCA Component 1 ({explained_variance_hidden[0]:.2f}% Variance)")
    axes[0].set_ylabel(f"PCA Component 2 ({explained_variance_hidden[1]:.2f}% Variance)")
    axes[0].legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

    sns.scatterplot(x=cell_pca[:, 0], y=cell_pca[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=axes[1])
    axes[1].set_title(f"LSTM Cell State PCA\nPC1={explained_variance_cell[0]:.2f}%, PC2={explained_variance_cell[1]:.2f}%")
    axes[1].set_xlabel(f"PCA Component 1 ({explained_variance_cell[0]:.2f}% Variance)")
    axes[1].set_ylabel(f"PCA Component 2 ({explained_variance_cell[1]:.2f}% Variance)")
    axes[1].legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x=hidden_tsne[:, 0], y=hidden_tsne[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=axes[0])
    axes[0].set_title("LSTM Hidden State t-SNE Projection")
    axes[0].set_xlabel("t-SNE Component 1")
    axes[0].set_ylabel("t-SNE Component 2")
    axes[0].legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

    sns.scatterplot(x=cell_tsne[:, 0], y=cell_tsne[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=axes[1])
    axes[1].set_title("LSTM Cell State t-SNE Projection")
    axes[1].set_xlabel("t-SNE Component 1")
    axes[1].set_ylabel("t-SNE Component 2")
    axes[1].legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    corr_hidden, _ = pearsonr(hidden_states_np.mean(axis=1), df_clean["Event_PupilDilation"])
    corr_cell, _ = pearsonr(cell_states_np.mean(axis=1), df_clean["Event_PupilDilation"])
    
    print(f"Pearson Correlation with Actual Pupil Dilation:")
    print(f"Hidden State Mean: {corr_hidden:.3f}")
    print(f"Cell State Mean: {corr_cell:.3f}")
    

def pca_lcne(model, X_tensor, df_clean):
    """
    For the Vanilla LC feedforward models, extracts activations and performs PCA, t-SNE, and applies clustering to visualize.
    """

    with torch.no_grad():
        prev_LC = torch.zeros(X_tensor.shape[0], model.hidden_dim)
        prev_Cortex = torch.zeros(X_tensor.shape[0], model.hidden_dim)
        
        LC_act, NE_act, C_act, Pupil_pred, LC_raw, NE_raw, C_raw = model(X_tensor, prev_LC, prev_Cortex, return_activations=True)

    act_lc = LC_act.cpu().numpy()
    act_ne = NE_act.cpu().numpy()
    act_cortex = C_act.cpu().numpy()

    act_combined = np.hstack([act_lc, act_ne, act_cortex])
    
    scaler = StandardScaler()
    act_combined_scaled = scaler.fit_transform(act_combined)
    pca = PCA(n_components=2)
    act_pca = pca.fit_transform(act_combined_scaled)
    explained_variance = pca.explained_variance_ratio_ * 100

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    act_tsne = tsne.fit_transform(act_pca)

    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters_pca = kmeans.fit_predict(act_pca)
    clusters_tsne = kmeans.predict(act_tsne)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    activations_list = [act_lc, act_ne, act_cortex]
    labels = ["LC", "NE", "Cortex"]

    for i, (activation, label) in enumerate(zip(activations_list, labels)):
        pca = PCA(n_components=2)
        act_pca = pca.fit_transform(activation)
        explained_variance = pca.explained_variance_ratio_ * 100
        
        # PCA Projection
        sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=axes[i])
        axes[i].set_title(f"{label} PCA\nExplained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")
        axes[i].set_xlabel(f"PCA Component 1 ({explained_variance[0]:.2f}% Variance)")
        axes[i].set_ylabel(f"PCA Component 2 ({explained_variance[1]:.2f}% Variance)")

    # K-Means Clustering
    sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=clusters_pca, palette="tab10", alpha=0.7, ax=axes[3])
    axes[3].set_title("PCA Clustering")

    sns.scatterplot(x=act_tsne[:, 0], y=act_tsne[:, 1], hue=clusters_tsne, palette="tab10", alpha=0.7, ax=axes[4])
    axes[4].set_title("t-SNE Clustering")

    plt.tight_layout()
    plt.show()


def pca_lcne_lstm(model, X_tensor, df_clean):
    '''PCA analysis for LCNE LSTM model'''
    model.eval()
    with torch.no_grad():
        prev_LC = torch.zeros(X_tensor.shape[0], model.hidden_dim)
        prev_Cortex = torch.zeros(X_tensor.shape[0], model.hidden_dim)
        cell_state = torch.zeros(X_tensor.shape[0], model.hidden_dim)

        LC_act, NE_act, C_act, Pupil_pred, forget_gate, input_gate, output_gate, cell_state = model(
            X_tensor, prev_LC, prev_Cortex, cell_state, return_activations=True
        )

    act_lc = LC_act.cpu().numpy()
    act_ne = NE_act.cpu().numpy()
    act_cortex = C_act.cpu().numpy()
    pupil_pred = Pupil_pred.cpu().numpy().squeeze()
    forget_gate_np = forget_gate.cpu().numpy()
    input_gate_np = input_gate.cpu().numpy()
    output_gate_np = output_gate.cpu().numpy()
    cell_state_np = cell_state.cpu().numpy()

    df_activations = pd.DataFrame({
        'LC_Mean': act_lc.mean(axis=1), 'LC_Var': act_lc.var(axis=1),
        'NE_Mean': act_ne.mean(axis=1), 'NE_Var': act_ne.var(axis=1),
        'Cortex_Mean': act_cortex.mean(axis=1), 'Cortex_Var': act_cortex.var(axis=1),
        'ForgetGate_Mean': forget_gate_np.mean(axis=1), 'ForgetGate_Var': forget_gate_np.var(axis=1),
        'InputGate_Mean': input_gate_np.mean(axis=1), 'InputGate_Var': input_gate_np.var(axis=1),
        'OutputGate_Mean': output_gate_np.mean(axis=1), 'OutputGate_Var': output_gate_np.var(axis=1),
        'CellState_Mean': cell_state_np.mean(axis=1), 'CellState_Var': cell_state_np.var(axis=1),
        'PupilPred': pupil_pred.mean(axis=1),
    })
    
    print (act_lc.mean(axis=1).shape, input_gate_np.mean(axis=1).shape, pupil_pred.shape)
    
    activations_list = [act_lc, act_ne, act_cortex, input_gate_np, output_gate_np, cell_state_np]
    labels = ["LC", "NE", "Cortex", "Input Gate", "Output Gate", "Cell State"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    for i, (activation, label) in enumerate(zip(activations_list, labels)):
        pca = PCA(n_components=2)
        act_pca = pca.fit_transform(activation)
        explained_variance = pca.explained_variance_ratio_ * 100
        
        ax = axes[i]
        sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=ax)
        ax.set_title(f"{label}\nExplained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")
        ax.set_xlabel(f"PCA Component 1 ({explained_variance[0]:.2f}% Variance)")
        ax.set_ylabel(f"PCA Component 2 ({explained_variance[1]:.2f}% Variance)")
        ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def analyze_ff_gadget_activations(model, X_tensor, df_clean):
    """
    Runs PCA for layer-wise activations in FFGadgetController and analyzes correlation,
    while also testing for an inverted-U relationship in NE firing.
    """
    model.eval()
    with torch.no_grad():
        Pupil_pred, LC_act, NE_act, tonic_NE, phasic_NE, hidden_1, hidden_2 = model(X_tensor, activation=True)

    activations_dict = {
        "LC": LC_act.cpu().numpy(),
        "NE": NE_act.cpu().numpy(),
        "Tonic_NE": tonic_NE.cpu().numpy(),
        "Phasic_NE": phasic_NE.cpu().numpy(),
        "Layer 1": hidden_1.cpu().numpy(),
        "Layer 2": hidden_2.cpu().numpy(),
    }

    pupil_pred = Pupil_pred.cpu().numpy().squeeze()
    pupil_actual = df_clean["Event_PupilDilation"].values

    # Ensure matching lengths
    min_length = min(len(pupil_actual), len(pupil_pred))
    pupil_actual, pupil_pred = pupil_actual[:min_length], pupil_pred[:min_length]

    df_activations = pd.DataFrame({
        f"{key}_Mean": act.mean(axis=1) for key, act in activations_dict.items()
    })
    df_activations["PupilPred"] = pupil_pred
    df_activations["ActualPupil"] = pupil_actual

    # --- PCA Analysis ---
    num_activations = len(activations_dict)
    cols = 3
    rows = -(-num_activations // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for i, (label, activation) in enumerate(activations_dict.items()):
        pca = PCA(n_components=2)
        act_pca = pca.fit_transform(activation)
        explained_variance = pca.explained_variance_ratio_ * 100

        sns.scatterplot(
            x=act_pca[:, 0], y=act_pca[:, 1], hue=df_clean["Condition"],
            palette="viridis", alpha=0.7, ax=axes[i]
        )
        axes[i].set_title(f"{label} Activations (PCA)\nPC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")
        axes[i].set_xlabel(f"PC1 ({explained_variance[0]:.2f}% Variance)")
        axes[i].set_ylabel(f"PC2 ({explained_variance[1]:.2f}% Variance)")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # --- Checking Inverted-U Hypothesis (Separate Plots) ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot Tonic NE ---
    sns.scatterplot(x=pupil_actual, y=tonic_NE.mean(axis=1), label="Tonic NE", alpha=0.5, ax=ax[0])
    
    poly = PolynomialFeatures(degree=2)
    pupil_poly = poly.fit_transform(pupil_actual.reshape(-1, 1))
    tonic_model = LinearRegression().fit(pupil_poly, tonic_NE.mean(axis=1))
    x_vals = np.linspace(pupil_actual.min(), pupil_actual.max(), 100)
    x_poly = poly.transform(x_vals.reshape(-1, 1))
    ax[0].plot(x_vals, tonic_model.predict(x_poly), color="blue", label="Tonic NE (Quadratic Fit)")
    
    ax[0].set_xlabel("Arousal Level (Pupil Dilation)")
    ax[0].set_ylabel("Tonic NE Activation")
    ax[0].set_title("Tonic NE vs. Arousal (Checking Inverted-U)")
    ax[0].legend()

    # --- Plot Phasic NE ---
    sns.scatterplot(x=pupil_actual, y=phasic_NE.mean(axis=1), label="Phasic NE", alpha=0.5, ax=ax[1])
    
    phasic_model = LinearRegression().fit(pupil_poly, phasic_NE.mean(axis=1))
    ax[1].plot(x_vals, phasic_model.predict(x_poly), color="red", label="Phasic NE (Quadratic Fit)")
    
    ax[1].set_xlabel("Arousal Level (Pupil Dilation)")
    ax[1].set_ylabel("Phasic NE Activation")
    ax[1].set_title("Phasic NE vs. Arousal (Checking Inverted-U)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    
    # --- Correlations ---
    correlations = {
        key: pearsonr(df_activations[f"{key}_Mean"], df_activations["ActualPupil"])[0]
        for key in activations_dict.keys()
    }
    correlations["Predicted Pupil Dilation"] = pearsonr(df_activations["PupilPred"], df_activations["ActualPupil"])[0]

    print("\n Pearson Correlation with Actual Pupil Dilation:")
    for key, value in correlations.items():
        print(f"{key}: {value:.3f}")


def evaluate_ff_uncertainty_gadget(model, X_test, Y_test, scaler_Y):
    """Evaluates the FF Gadget Controller on pupil dilation prediction + uncertainty."""
    
    model.eval()
    with torch.no_grad():
        pupil_mean, pupil_var = model(X_test)

        Y_pupil_actual = Y_test.cpu().numpy()

        pupil_pred = pupil_mean.cpu().numpy().squeeze()
        pupil_uncertainty = np.sqrt(pupil_var.cpu().numpy().squeeze())  # Convert variance to std dev

    pupil_pred_rescaled = scaler_Y.inverse_transform(pupil_pred.reshape(-1, 1)).squeeze()
    pupil_actual_rescaled = scaler_Y.inverse_transform(Y_pupil_actual.reshape(-1, 1)).squeeze()

    # Compute Error Metrics
    pupil_mae = np.mean(np.abs(pupil_pred_rescaled - pupil_actual_rescaled))
    pupil_mse = np.mean((pupil_pred_rescaled - pupil_actual_rescaled) ** 2)

    print(f"Pupil Dilation MAE: {pupil_mae:.4f}")
    print(f"Pupil Dilation MSE: {pupil_mse:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(pupil_actual_rescaled, pupil_pred_rescaled, label="Predictions", alpha=0.6)
    plt.fill_between(
        pupil_actual_rescaled,
        pupil_pred_rescaled - 2 * pupil_uncertainty,
        pupil_pred_rescaled + 2 * pupil_uncertainty,
        color='blue', alpha=0.2, label="±2σ Uncertainty"
    )
    plt.xlabel("Actual Pupil Dilation")
    plt.ylabel("Predicted Pupil Dilation")
    plt.title("Pupil Dilation Predictions with Uncertainty")
    plt.legend()
    plt.show()


# ===========================================================================
# Topological analaysis helper functions
# ===========================================================================


def extract_activations_gadget(model, X_tensor):
    """Extract activations from the model."""
    model.eval()
    with torch.no_grad():
        if isinstance(model, FFGadgetUncertainController):
            pupil_mean, pupil_var, LC_t, NE_t, tonic_NE, phasic_NE, hidden_1, hidden_2 = model(X_tensor, activation=True)
        elif isinstance(model, FFGadgetController):
            pupil_mean, LC_t, NE_t, tonic_NE, phasic_NE, hidden_1, hidden_2 = model(X_tensor, activation=True)
        else:
            raise ValueError(f"Unsupported model type: {model}")
        
    activations_dict = {
        "LC": LC_t.cpu().numpy(),
        "NE": NE_t.cpu().numpy(),
        "Tonic_NE": tonic_NE.cpu().numpy(),
        "Phasic_NE": phasic_NE.cpu().numpy(),
        "Hidden_1": hidden_1.cpu().numpy(),
        "Hidden_2": hidden_2.cpu().numpy(),
    }

    return activations_dict


def compute_persistence_distance(activations_neutral, activations_stress):
    """Computes Wasserstein distance between persistence diagrams."""
    diagrams_neutral = ripser(activations_neutral)['dgms']
    diagrams_stress = ripser(activations_stress)['dgms']

    dist_h0 = wasserstein(diagrams_neutral[0], diagrams_stress[0])
    dist_h1 = wasserstein(diagrams_neutral[1], diagrams_stress[1])

    print(f"Wasserstein Distance (H0): {dist_h0:.4f}")
    print(f"Wasserstein Distance (H1): {dist_h1:.4f}")
    
    return dist_h0, dist_h1


def compute_persistence_distance_df(activations_dict):
    """Computes Wasserstein distances between conditions for each shared key and stores them in a dataframe."""
    results = []
    
    conditions = list(activations_dict.keys())
    shared_keys = set(activations_dict[conditions[0]].keys())

    for key in shared_keys:
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                cond1, cond2 = conditions[i], conditions[j]
                act1, act2 = activations_dict[cond1][key], activations_dict[cond2][key]

                act1 = np.asarray(act1)
                act2 = np.asarray(act2)

                diagrams1 = ripser(act1)['dgms']
                diagrams2 = ripser(act2)['dgms']

                dist_h0 = wasserstein(diagrams1[0], diagrams2[0])
                dist_h1 = wasserstein(diagrams1[1], diagrams2[1])

                results.append({
                    "Key": key,
                    "Condition 1": cond1,
                    "Condition 2": cond2,
                    "Wasserstein H0": dist_h0,
                    "Wasserstein H1": dist_h1
                })
    
    df_results = pd.DataFrame(results)
    return df_results


def wasserstein_permutation_test(act1, act2, n_permutations=1000):
    """Runs a permutation test on Wasserstein distance between two persistence diagrams."""
    
    def wasserstein_stat(x, y):
        return wasserstein(x.reshape(-1, 2), y.reshape(-1, 2))

    # Compute persistence diagrams
    diagrams1 = ripser(np.asarray(act1))['dgms']
    diagrams2 = ripser(np.asarray(act2))['dgms']

    def format_diagram(diag):
        """Ensure persistence diagram is valid and remove inf values."""
        diag = np.asarray(diag)
        diag = diag[~np.isinf(diag).any(axis=1)]  # Remove rows with `inf`
        if diag.ndim == 1 or len(diag) == 0:  # If it's 1D or empty
            return np.zeros((1, 2))  # Placeholder
        return diag

    # Ensure diagrams are 2D and finite
    diagrams1 = [format_diagram(d) for d in diagrams1]
    diagrams2 = [format_diagram(d) for d in diagrams2]

    # print("Filtered Diagrams1 Shapes:", [np.array(d).shape for d in diagrams1])
    # print("Filtered Diagrams2 Shapes:", [np.array(d).shape for d in diagrams2])

    # Compute Wasserstein distance
    dist_h0 = wasserstein(diagrams1[0], diagrams2[0])
    dist_h1 = wasserstein(diagrams1[1], diagrams2[1])

    # Flatten diagrams for permutation test
    flat_diag1_h0 = diagrams1[0].flatten()
    flat_diag2_h0 = diagrams2[0].flatten()
    flat_diag1_h1 = diagrams1[1].flatten()
    flat_diag2_h1 = diagrams2[1].flatten()

    # Run permutation test on **filtered, finite** data
    result_h0 = permutation_test((flat_diag1_h0, flat_diag2_h0), wasserstein_stat, n_resamples=n_permutations)
    result_h1 = permutation_test((flat_diag1_h1, flat_diag2_h1), wasserstein_stat, n_resamples=n_permutations)

    return {
        "Wasserstein H0": dist_h0, "p-value H0": result_h0.pvalue,
        "Wasserstein H1": dist_h1, "p-value H1": result_h1.pvalue
    }
    

def compute_persistence_hypothesis_test(activations_dict, n_permutations=1000, check_for=['NE']):
    """Computes Wasserstein distances and runs a hypothesis test between conditions."""
    results = []
    
    conditions = list(activations_dict.keys())
    shared_keys = set(activations_dict[conditions[0]].keys())

    for key in check_for:
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                cond1, cond2 = conditions[i], conditions[j]
                act1, act2 = activations_dict[cond1][key], activations_dict[cond2][key]

                test_results = wasserstein_permutation_test(act1, act2, n_permutations=n_permutations)

                results.append({
                    "Key": key,
                    "Condition 1": cond1,
                    "Condition 2": cond2,
                    "Wasserstein H0": test_results["Wasserstein H0"],
                    "p-value H0": test_results["p-value H0"],
                    "Wasserstein H1": test_results["Wasserstein H1"],
                    "p-value H1": test_results["p-value H1"],
                })
    
    df_results = pd.DataFrame(results)
    return df_results


def compute_persistence_lifetimes(diagram):
    """Computes lifetimes (Death - Birth) for persistence features."""
    return diagram[:, 1] - diagram[:, 0]


def compare_persistence_lifetimes(diagrams1, diagrams2, condition1="Neutral", condition2="Stressful"):
    """
    Compares persistence lifetimes using Kolmogorov-Smirnov (KS) and t-test.
    Assumes `diagrams1` and `diagrams2` are lists of persistence diagrams (e.g., [H0, H1]).
    """
    results = []
    
    for dim in range(len(diagrams1)):  # H0 and H1
        lifetimes1 = compute_persistence_lifetimes(diagrams1[dim])
        lifetimes2 = compute_persistence_lifetimes(diagrams2[dim])
        
        # remove infinite points if any exist
        lifetimes1 = lifetimes1[np.isfinite(lifetimes1)]
        lifetimes2 = lifetimes2[np.isfinite(lifetimes2)]
        
        # KS test
        ks_stat, ks_pval = stats.ks_2samp(lifetimes1, lifetimes2)
        
        # t-test (Welch’s t-test for unequal variances)
        t_stat, t_pval = stats.ttest_ind(lifetimes1, lifetimes2, equal_var=False)

        results.append({
            "Dimension": f"H{dim}",
            "Condition 1": condition1,
            "Condition 2": condition2,
            "Mean Lifetime 1": np.mean(lifetimes1),
            "Mean Lifetime 2": np.mean(lifetimes2),
            "KS Stat": ks_stat,
            "KS p-value": ks_pval,
            "t-test Stat": t_stat,
            "t-test p-value": t_pval,
        })

        plt.figure(figsize=(6, 4))
        plt.hist(lifetimes1, bins=20, alpha=0.6, label=condition1, density=True)
        plt.hist(lifetimes2, bins=20, alpha=0.6, label=condition2, density=True)
        plt.xlabel("Persistence Lifetime")
        plt.ylabel("Density")
        plt.title(f"Persistence Lifetime Distribution ({condition1} vs. {condition2}) - H{dim}")
        plt.legend()
        plt.show()

    df_results = pd.DataFrame(results)
    return df_results


def plot_birth_death_distributions(activations_dict, conditions=("Neutral", "Stressful"), activation_name='LC', homology_dim=0):
    """
    Plots the KDE of birth and death times for given conditions and homology dimension.
    """

    birth_times = {}
    death_times = {}

    for cond in conditions:
        activations = np.asarray(activations_dict[cond][activation_name])
        diagrams = ripser(activations)['dgms']

        # extract birth and death values for the given homology dimension
        births, deaths = diagrams[homology_dim][:, 0], diagrams[homology_dim][:, 1]

        birth_times[cond] = births
        death_times[cond] = deaths

    # KDE for Birth Times
    plt.figure(figsize=(8, 4))
    for cond in conditions:
        sns.kdeplot(birth_times[cond], label=f"{cond} - Birth", shade=True)
    plt.xlabel("Birth Time")
    plt.ylabel("Density")
    plt.title(f"Birth Time Distributions (H{homology_dim})")
    plt.legend()
    plt.show()

    # KDE for Death Times
    plt.figure(figsize=(8, 4))
    for cond in conditions:
        sns.kdeplot(death_times[cond], label=f"{cond} - Death", shade=True)
    plt.xlabel("Death Time")
    plt.ylabel("Density")
    plt.title(f"Death Time Distributions (H{homology_dim})")
    plt.legend()
    plt.show()


def hypothesis_test_persistence_distributions(activations_dict, conditions=("Neutral", "Stressful"), activation_name='LC', homology_dim=0):
    """
    Performs statistical tests on birth and death time distributions between conditions.
    """

    birth_times = {}
    death_times = {}

    for cond in conditions:
        activations = np.asarray(activations_dict[cond][activation_name])
        diagrams = ripser(activations)['dgms']

        births, deaths = diagrams[homology_dim][:, 0], diagrams[homology_dim][:, 1]

        birth_times[cond] = births
        death_times[cond] = deaths

    results = []

    def test_distributions(data1, data2, label):
        ks_stat, ks_pval = ks_2samp(data1, data2)
        mw_stat, mw_pval = mannwhitneyu(data1, data2, alternative="two-sided")

        results.append({
            "Metric": label,
            "KS Test Stat": ks_stat, "KS p-value": ks_pval,
            "MW Test Stat": mw_stat, "MW p-value": mw_pval
        })

    if len(birth_times[conditions[0]]) > 0 and len(birth_times[conditions[1]]) > 0:
        test_distributions(birth_times[conditions[0]], birth_times[conditions[1]], f"Birth (H{homology_dim})")

    if len(death_times[conditions[0]]) > 0 and len(death_times[conditions[1]]) > 0:
        test_distributions(death_times[conditions[0]], death_times[conditions[1]], f"Death (H{homology_dim})")

    print(f"\nHypothesis Test Results For Homology H{homology_dim}:")
    for res in results:
        print(f"{res['Metric']} → KS p={res['KS p-value']:.4f}, MW p={res['MW p-value']:.4f}")

    return results


def compute_persistent_homology(activation_data, title="Persistent Homology"):
    """Computes persistent homology and plots the persistence diagram."""
    diagrams = ripser(activation_data)['dgms']  # Compute persistence diagram
    plot_diagrams(diagrams, show=True)


def compute_persistent_homology_overlay(act_neutral, act_stress, title="Persistent Homology Overlay"):
    """Computes and overlays persistent homology for neutral vs. stressful activations using Ripser."""
    
    diagrams_neutral = ripser(act_neutral)['dgms']
    diagrams_stress = ripser(act_stress)['dgms']

    # Colors and markers for different homology dimensions
    homology_styles = {
        0: {"color": "blue", "marker": "o", "label": r"$H_0$ (Neutral)", "label_stress": r"$H_0$ (Stressful)"},
        1: {"color": "pink", "marker": "s", "label": r"$H_1$ (Neutral)", "label_stress": r"$H_1$ (Stressful)"},
        2: {"color": "green", "marker": "D", "label": r"$H_2$ (Neutral)", "label_stress": r"$H_2$ (Stressful)"},
    }

    plt.figure(figsize=(6, 6))

    # Iterate over dimensions and plot
    for dim in range(len(diagrams_neutral)):
        if dim in homology_styles:
            style = homology_styles[dim]
            if len(diagrams_neutral[dim]) > 0:
                plt.scatter(diagrams_neutral[dim][:, 0], diagrams_neutral[dim][:, 1], 
                            c=style["color"], marker=style["marker"], label=style["label"])
            if len(diagrams_stress[dim]) > 0:
                plt.scatter(diagrams_stress[dim][:, 0], diagrams_stress[dim][:, 1], 
                            c=style["color"], marker=style["marker"], edgecolors="black", label=style["label_stress"])

    # Diagonal reference line
    plt.plot([0, 2], [0, 2], "k--", alpha=0.5)  

    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title(title)
    plt.legend()
    plt.show()
    

def compute_mapper_graph(activation_data, n_neighbors=10, title="Mapper Graph"):
    """Computes and visualizes a Mapper graph using K-Nearest Neighbors (KNN)."""
    
    # reduce dimension for visualization
    pca = PCA(n_components=2)
    low_dim_data = pca.fit_transform(activation_data)

    # compute KNN graph
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(low_dim_data)
    distances, indices = knn.kneighbors(low_dim_data)

    # create graph
    G = nx.Graph()
    for i in range(len(low_dim_data)):
        for j in indices[i]:
            if i != j:
                G.add_edge(i, j, weight=distances[i, np.where(indices[i] == j)[0][0]])

    # Mapper graph
    plt.figure(figsize=(7, 6))
    nx.draw(G, pos={i: low_dim_data[i] for i in range(len(low_dim_data))}, node_size=30, edge_color='gray')
    plt.title(title)
    plt.show()


def compute_mapper_graph_overlay(act_neutral, act_stress, n_neighbors=10, title="Mapper Graph Overlay"):
    """Computes and overlays Mapper Graphs for neutral vs. stressful activations using KNN."""
    
    pca = PCA(n_components=2)
    low_dim_neutral = pca.fit_transform(act_neutral)
    low_dim_stress = pca.fit_transform(act_stress)

    knn_neutral = NearestNeighbors(n_neighbors=n_neighbors).fit(low_dim_neutral)
    knn_stress = NearestNeighbors(n_neighbors=n_neighbors).fit(low_dim_stress)

    distances_neutral, indices_neutral = knn_neutral.kneighbors(low_dim_neutral)
    distances_stress, indices_stress = knn_stress.kneighbors(low_dim_stress)

    G_neutral = nx.Graph()
    for i in range(len(low_dim_neutral)):
        for j in indices_neutral[i]:
            if i != j:
                G_neutral.add_edge(i, j, weight=distances_neutral[i, np.where(indices_neutral[i] == j)[0][0]])

    G_stress = nx.Graph()
    for i in range(len(low_dim_stress)):
        for j in indices_stress[i]:
            if i != j:
                G_stress.add_edge(i, j, weight=distances_stress[i, np.where(indices_stress[i] == j)[0][0]])

    plt.figure(figsize=(7, 6))
    nx.draw(G_neutral, pos={i: low_dim_neutral[i] for i in range(len(low_dim_neutral))}, node_size=30, edge_color='blue', alpha=0.6, label="Neutral")
    nx.draw(G_stress, pos={i: low_dim_stress[i] for i in range(len(low_dim_stress))}, node_size=30, edge_color='pink', alpha=0.6, label="Stressful")

    plt.title(title)
    plt.legend(["Neutral", "Stressful"])
    plt.show()


def plot_manifold_projection(activations, labels, method="tsne"):
    """Visualizes activation embeddings using t-SNE or UMAP."""
    
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method should be 'tsne' or 'umap'")

    embeddings = reducer.fit_transform(activations)
    
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, palette="flare", alpha=0.7)
    plt.title(f"{method.upper()} Projection of Activations")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


def construct_activation_graph(activations, threshold=0.7):
    """Constructs a graph where nodes are activations and edges are high-correlation pairs."""
    
    # similarity matrix
    similarity_matrix = cosine_similarity(activations)

    # create a graph
    G = nx.Graph()
    num_nodes = similarity_matrix.shape[0]

    # add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # add edges based on threshold
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    return G


def plot_activation_graph(G, title="Activation Graph"):
    """Plots a simple network graph of activations."""
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=50, edge_color="gray", alpha=0.7)
    plt.title(title)
    plt.show()
    

def compute_betti_curves(activations_dict, homology_dim=1,  activation_name='LC'):
    """Computes Betti curves for different conditions."""
    plt.figure(figsize=(7, 5))

    for cond, activations in activations_dict.items():
        diagrams = ripser(np.asarray(activations[activation_name]))["dgms"]
        birth_times = diagrams[homology_dim][:, 0]
        death_times = diagrams[homology_dim][:, 1]

        bins = np.linspace(min(birth_times), max(death_times), 50)
        betti_counts = np.array([(birth_times < t).sum() - (death_times < t).sum() for t in bins])
        plt.plot(bins, betti_counts, label=f"{cond} (H{homology_dim})")

    plt.xlabel("Filtration Value")
    plt.ylabel("Betti Number Count")
    plt.title(f"Betti Curves (H{homology_dim})")
    plt.legend()
    plt.show()
    

def compute_topological_silhouette(activations_dict,homology_dim=1, activation_name='LC'):
    """Computes topological silhouette across conditions."""
    plt.figure(figsize=(7, 5))
    
    for cond, activations in activations_dict.items():
        diagrams = ripser(np.asarray(activations[activation_name]))["dgms"]
        lifetimes = diagrams[homology_dim][:, 1] - diagrams[homology_dim][:, 0]
        weights = lifetimes / np.sum(lifetimes)
        plt.plot(diagrams[homology_dim][:, 0], weights, label=f"{cond} (H{homology_dim})", alpha=0.7)
    
    plt.xlabel("Birth")
    plt.ylabel("Persistence Weight")
    plt.title(f"Topological Silhouette (H{homology_dim})")
    plt.legend()
    plt.show()


# ===========================================================================
# Experimental Analysis
# ===========================================================================


def analyze_uncertainty_relationships(model, X_test):
    """
    Extracts LC-NE activations, pupil dilation predictions, and uncertainty.
    Then, analyzes how LC activation, tonic NE, and phasic NE correlate with uncertainty.
    """

    # extract signals from the model
    model.eval()
    with torch.no_grad():
        pupil_mean, pupil_var, LC_t, NE_t, tonic_NE, phasic_NE, _, _ = model(X_test, activation=True)

    pupil_mean = pupil_mean.cpu().numpy()
    pupil_var = pupil_var.cpu().numpy()
    LC_t = LC_t.cpu().numpy()
    NE_t = NE_t.cpu().numpy()
    tonic_NE = tonic_NE.cpu().numpy()
    phasic_NE = phasic_NE.cpu().numpy()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=LC_t.mean(axis=1), y=pupil_var.squeeze(), alpha=0.5)
    plt.xlabel("Mean LC Activation")
    plt.ylabel("Predicted Uncertainty (Pupil Variance)")
    plt.title("LC Activation vs. Predicted Uncertainty")
    plt.show()

    # compare tonic vs. phasic NE with uncertainty
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(x=tonic_NE.mean(axis=1), y=pupil_var.squeeze(), alpha=0.5, ax=ax[0])
    ax[0].set_xlabel("Tonic NE Activation")
    ax[0].set_ylabel("Predicted Uncertainty")
    ax[0].set_title("Tonic NE vs. Uncertainty")

    sns.scatterplot(x=phasic_NE.mean(axis=1), y=pupil_var.squeeze(), alpha=0.5, ax=ax[1])
    ax[1].set_xlabel("Phasic NE Activation")
    ax[1].set_ylabel("Predicted Uncertainty")
    ax[1].set_title("Phasic NE vs. Uncertainty")

    plt.tight_layout()
    plt.show()

    results = {
        "LC vs. Uncertainty": pearsonr(LC_t.mean(axis=1), pupil_var.squeeze())[0],
        "Tonic NE vs. Uncertainty": pearsonr(tonic_NE.mean(axis=1), pupil_var.squeeze())[0],
        "Phasic NE vs. Uncertainty": pearsonr(phasic_NE.mean(axis=1), pupil_var.squeeze())[0],
        "NE vs. Uncertainty": pearsonr(NE_t.mean(axis=1), pupil_var.squeeze())[0],
    }
    
    print("\n📊 **Correlation Results:**")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    return results

def simulate_lc_activation(model, X_test, lc_boost=1.0):
    """Artificially boost or suppress LC activation and analyze effects on uncertainty."""
    model.eval()
    with torch.no_grad():
        pupil_high, var_high = model(X_test + lc_boost)  # High LC activation
        pupil_low, var_low = model(X_test - lc_boost)  # Low LC activation

    return pupil_high.cpu().numpy(), var_high.cpu().numpy(), pupil_low.cpu().numpy(), var_low.cpu().numpy()


def compute_autocorrelation(activations_dict, conditions=("Neutral", "Stressful"), activation_name='LC', lags=20):
    """
    Computes auto-correlation and partial auto-correlation for activations.
    """
    plt.figure(figsize=(12, 5))
    
    for i, cond in enumerate(conditions):
        activations = np.asarray(activations_dict[cond][activation_name]).mean(axis=1)
        
        acf_vals = acf(activations, nlags=lags)
        pacf_vals = pacf(activations, nlags=lags)
        
        plt.subplot(1, 2, 1)
        plt.plot(acf_vals, label=cond)
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.title("Auto-Correlation Function")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(pacf_vals, label=cond)
        plt.xlabel("Lag")
        plt.ylabel("PACF")
        plt.title("Partial Auto-Correlation Function")
        plt.legend()

    plt.show()

def compute_activation_entropy(activations_dict, conditions=("Neutral", "Stressful"), activation_name='LC'):
    """
    Computes entropy and KL-Divergence between activation distributions.
    """
    distributions = {}
    for cond in conditions:
        activations = np.asarray(activations_dict[cond][activation_name]).flatten()
        hist, bins = np.histogram(activations, bins=30, density=True)
        distributions[cond] = hist

    entropy_vals = {cond: entropy(distributions[cond]) for cond in conditions}
    kl_div = entropy(distributions[conditions[0]], distributions[conditions[1]])

    print("\nEntropy & Divergence Results:")
    for cond, ent in entropy_vals.items():
        print(f"{cond} Entropy: {ent:.4f}")

    print(f"KL-Divergence ({conditions[0]} → {conditions[1]}): {kl_div:.4f}")

    return entropy_vals, kl_div
