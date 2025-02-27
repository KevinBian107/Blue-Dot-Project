import pandas as pd
import numpy as np
import torch

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns

from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors

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


def extract_activations(model, X_tensor):
    """Extract activations from the model."""
    model.eval()
    with torch.no_grad():
        pupil_mean, pupil_var, LC_t, NE_t, tonic_NE, phasic_NE, hidden_1, hidden_2 = model(X_tensor, activation=True)

    activations_dict = {
        "LC": LC_t.cpu().numpy(),
        "NE": NE_t.cpu().numpy(),
        "Tonic_NE": tonic_NE.cpu().numpy(),
        "Phasic_NE": phasic_NE.cpu().numpy(),
        "Hidden_1": hidden_1.cpu().numpy(),
        "Hidden_2": hidden_2.cpu().numpy(),
    }

    return activations_dict

def compute_persistent_homology(activation_data, title="Persistent Homology"):
    """Computes persistent homology and plots the persistence diagram."""
    diagrams = ripser(activation_data)['dgms']  # Compute persistence diagram
    plot_diagrams(diagrams, show=True)

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