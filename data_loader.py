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
    
    # df_clean = df_clean[df_clean['Condition'] == 'NEUTRAL']

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


def preprocess_data_stress(df_behavior, encoder=None, scaler_X=None, condition="AROUSING"):
    '''Preprocess behavioral data for stress condition while keeping feature dimensions consistent.'''
    
    features = ['Condition', 'PreEvent_PupilMax', 'TrialEvent', 'onset', 'duration']
    target = ['Event_PupilDilation']

    df_clean_all = df_behavior[features + target].dropna().reset_index(drop=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df_clean_all[['Condition', 'TrialEvent']])  # Fit on all conditions
    encoded_feature_names = encoder.get_feature_names_out(['Condition', 'TrialEvent'])

    scaler_X = StandardScaler()
    scaler_X.fit(df_clean_all[['PreEvent_PupilMax', 'onset', 'duration']])

    # filter only stressful data (Ensures no feature dimension mismatch)
    if condition == "AROUSING":
        df_clean_stress = df_clean_all[df_clean_all['Condition'] == 'AROUSING']
    elif condition == "NEUTRAL":
        df_clean_stress = df_clean_all[df_clean_all['Condition'] == 'NEUTRAL']

    scaled_features = scaler_X.transform(df_clean_stress[['PreEvent_PupilMax', 'onset', 'duration']])
    X_scaled = pd.DataFrame(scaled_features, columns=['PreEvent_PupilMax', 'onset', 'duration'])

    encoded_stressful = encoder.transform(df_clean_stress[['Condition', 'TrialEvent']])
    X_encoded = pd.DataFrame(encoded_stressful, columns=encoded_feature_names)

    X_scaled.reset_index(drop=True, inplace=True)
    X_encoded.reset_index(drop=True, inplace=True)
    X = pd.concat([X_scaled, X_encoded], axis=1)

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    
    return X_tensor
