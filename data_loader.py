import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import torch
import numpy as np

def load_participants_info(dataset_path):
    '''Load participants information'''
    
    participants_file = os.path.join(dataset_path, "participants.tsv")
    if os.path.exists(participants_file):
        df_participants = pd.read_csv(participants_file, sep='\t')
        # print("Participants Information:")
        # print(df_participants.head())
        return df_participants
    else:
        print("Participants file not found.")
        return None

def load_behavioral_data(dataset_path, participant_id):
    '''Load behavioral and pupil data from a participant's functional folder'''
    
    participant_folder = os.path.join(dataset_path, f"sub-{participant_id}", "func")
    
    if not os.path.exists(participant_folder):
        # print(f"Functional folder not found for participant {participant_id}")
        return None

    data_files = [f for f in os.listdir(participant_folder) if f.endswith(".tsv")]
    
    df_list = []
    for file in data_files:
        file_path = os.path.join(participant_folder, file)
        df = pd.read_csv(file_path, sep='\t')
        df["subject"] = participant_id
        df["task"] = file.split("_")[1]  # Extract task name
        df_list.append(df)
    
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        print(f"No .tsv files found for participant {participant_id}")
        return None

def load_event_descriptions(dataset_path):
    '''Load event descriptions from JSON file'''
    
    json_file = os.path.join(dataset_path, "task-Overlap_events.json")
    
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            event_descriptions = json.load(f)
            # print("Event Description:")
            # print(json.dumps(event_descriptions, indent=4))
            return event_descriptions
    else:
        print("Event description JSON not found.")
        return None

def preprocess_data(df_behavior):
    '''Preprocess behavioral data'''
    
    features = ['Condition', 'PreEvent_PupilMax', 'TrialEvent', 'onset', 'duration']
    target = ['Event_PupilDilation']

    df_clean = df_behavior[features + target].dropna().reset_index(drop=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df_clean[['Condition', 'TrialEvent']])
    encoded_feature_names = encoder.get_feature_names_out(['Condition', 'TrialEvent'])

    scaler_X = StandardScaler()
    scaled_features = scaler_X.fit_transform(df_clean[['PreEvent_PupilMax', 'onset', 'duration']])

    X_scaled = pd.DataFrame(scaled_features, columns=['PreEvent_PupilMax', 'onset', 'duration'])
    X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    X_scaled.reset_index(drop=True, inplace=True)
    X_encoded.reset_index(drop=True, inplace=True)
    X = pd.concat([X_scaled, X_encoded], axis=1)

    # scaler_Y = StandardScaler()
    # Y = scaler_Y.fit_transform(df_clean[['Event_PupilDilation']].values.reshape(-1,1))

    scaler_Y = MinMaxScaler(feature_range=(-1, 1))
    Y = scaler_Y.fit_transform(df_clean[['Event_PupilDilation']].values.reshape(-1, 1))

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).squeeze()

    print(f"X Shape: {X_tensor.shape}, Y Shape: {Y_tensor.shape}")
    print(f"Y Min: {Y_tensor.min().item()}, Y Max: {Y_tensor.max().item()}")  # Check Scaling
    
    return X, Y, X_tensor, Y_tensor, scaler_X, scaler_Y, df_clean


# def preprocess_data_multihead(df_behavior):
#     '''Preprocess behavioral data for FF Controller with multi-class memory strength.'''
    
#     # Features: Categorical + Continuous
#     categorical_features = ['Condition', 'TrialEvent']
#     continuous_features = ['PreEvent_PupilMax', 'onset', 'duration']
    
#     # Targets: Continuous (Pupil Dilation) & Categorical (Memory Strength)
#     target_continuous = ['Event_PupilDilation']
#     target_categorical = ['MemoryStrength']  # Multi-class classification

#     # Drop NaNs
#     df_clean = df_behavior[categorical_features + continuous_features + target_continuous + target_categorical].dropna().reset_index(drop=True)

#     # --- One-Hot Encoding for Categorical Features ---
#     encoder_features = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     encoded_features = encoder_features.fit_transform(df_clean[categorical_features])
#     encoded_feature_names = encoder_features.get_feature_names_out(categorical_features)

#     # --- One-Hot Encoding for Memory Strength (Multi-Class) ---
#     encoder_memory = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     encoded_memory = encoder_memory.fit_transform(df_clean[target_categorical])
#     encoded_memory_names = encoder_memory.get_feature_names_out(target_categorical)

#     # --- Standard Scaling for Continuous Features ---
#     scaler_X = StandardScaler()
#     scaled_features = scaler_X.fit_transform(df_clean[continuous_features])

#     # --- Prepare Feature DataFrame ---
#     X_scaled = pd.DataFrame(scaled_features, columns=continuous_features)
#     X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names)

#     X_scaled.reset_index(drop=True, inplace=True)
#     X_encoded.reset_index(drop=True, inplace=True)
#     X = pd.concat([X_scaled, X_encoded], axis=1)

#     # --- Scaling Continuous Targets ---
#     scaler_Y = MinMaxScaler(feature_range=(-1, 1))  # Normalize within [-1,1]
#     Y_cont = scaler_Y.fit_transform(df_clean[target_continuous])

#     # --- Convert Memory Strength (Multi-Class) to Tensor ---
#     Y_categorical = pd.DataFrame(encoded_memory, columns=encoded_memory_names)

#     # --- Convert to Tensors ---
#     X_tensor = torch.tensor(X.values, dtype=torch.float32)
#     Y_tensor = torch.tensor(np.hstack((Y_cont, Y_categorical)), dtype=torch.float32)  # Stack continuous + categorical

#     print(f"X Shape: {X_tensor.shape}, Y Shape: {Y_tensor.shape}")
#     print(f"Y Continuous Min: {Y_tensor[:, 0].min().item()}, Max: {Y_tensor[:, 0].max().item()}")  # Check Scaling

#     return X, Y_cont, X_tensor, Y_tensor, scaler_X, scaler_Y, df_clean
