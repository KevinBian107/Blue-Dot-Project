import torch
import torch.nn as nn

class UntrainedLCFCN(nn.Module):
    '''Untrained vanilla FCN based LCNE model'''
    def __init__(self, lambda_cortex):
        super(UntrainedLCFCN, self).__init__()
        
        self.lambda_cortex = lambda_cortex
        self.W_x = nn.Parameter(torch.tensor(0.5))  # Input weight to LC
        self.W_h = nn.Parameter(torch.tensor(0.7))  # LC recurrent weight
        self.W_LC = nn.Parameter(torch.tensor(0.8))  # LC-to-Cortex modulation
        self.W_C = nn.Parameter(torch.tensor(0.5))  # Stimulus-to-Cortex weight
        self.b = nn.Parameter(torch.tensor(0.1))  # LC bias

    def forward(self, x_t, stress_t, prev_LC, prev_Cortex):
        LC_t = torch.tanh(self.W_x * x_t + self.W_h * prev_LC + self.b + stress_t)
        
        NE_t = torch.sigmoid(self.W_LC * LC_t)
        
        C_t = prev_Cortex + self.lambda_cortex * self.W_LC * NE_t + self.W_C * x_t
        
        return LC_t, NE_t, C_t

class LCNECortexFitterBasic(nn.Module):
    '''Basic LCNECortex model with learnable parameters'''
    def __init__(self, lambda_cortex=0.1):
        super(LCNECortexFitter, self).__init__()
        self.lambda_cortex = lambda_cortex

        # learnable parameters to optimized for fiting real data, initialize with some values
        self.W_x = nn.Parameter(torch.tensor(0.5))  # Sensory input weight
        self.W_h = nn.Parameter(torch.tensor(0.7))  # LC recurrent weight
        self.W_LC = nn.Parameter(torch.tensor(0.8))  # LC-to-Cortex modulation
        self.W_C = nn.Parameter(torch.tensor(0.5))  # Sensory-to-Cortex weight
        self.W_LC_Pupil = nn.Parameter(torch.tensor(0.4))
        self.W_C_Pupil = nn.Parameter(torch.tensor(0.3))
        self.W_NE_Pupil = nn.Parameter(torch.tensor(0.2))
        self.b = nn.Parameter(torch.tensor(0.1))  # LC bias

    def forward(self, X, prev_LC, prev_Cortex):
        #  LC activity update for batch
        LC_t = torch.tanh(self.W_x * X[:, 0] + self.W_h * prev_LC + self.b)

        # NE release
        NE_t = torch.sigmoid(self.W_LC * LC_t)

        # cortical activity update
        C_t = prev_Cortex + self.lambda_cortex * self.W_LC * NE_t + self.W_C * X[:, 0]

        # pupil dilation (arousal effect)
        # Pupil_t = 0.4 * LC_t + 0.3 * C_t + 0.2 * NE_t + 0.1 * torch.sin(X[:, 0])
        Pupil_t = self.W_LC_Pupil * LC_t + self.W_C_Pupil * C_t + self.W_NE_Pupil * NE_t  # Learnable weighting

        return LC_t, NE_t, C_t, Pupil_t
    
class LCNECortexFitter(nn.Module):
    """Improved LCNECortex model with learnable parameters and batch normalization"""
    def __init__(self, input_dim, hidden_dim=8, lambda_cortex=0.1):
        super(LCNECortexFitter, self).__init__()
        self.lambda_cortex = lambda_cortex
        self.hidden_dim = hidden_dim

        # linear layers to correctly map dimensions
        self.W_x = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC = nn.Linear(hidden_dim, hidden_dim)
        self.W_C = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC_Pupil = nn.Linear(hidden_dim, 1)

        # batch normalization layers for stable training
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.relu = nn.LeakyReLU(0.1)  # prevents dead neurons

    def forward(self, X, prev_LC, prev_Cortex, return_activations=False):
        x_input = self.W_x(X)

        # LC activity
        LC_raw = x_input + self.W_h(prev_LC)  
        LC_t = self.relu(self.bn1(LC_raw))

        # NE release
        NE_raw = self.W_LC(LC_t)
        NE_t = self.relu(self.bn2(NE_raw))

        # cortical activity update
        C_raw = prev_Cortex + self.lambda_cortex * self.W_LC(NE_t) + self.W_C(x_input)
        C_t = self.relu(self.bn3(C_raw))

        # pupil dilation output
        Pupil_t = self.W_LC_Pupil(LC_t) + self.W_LC_Pupil(C_t) + self.W_LC_Pupil(NE_t)

        if return_activations:
            return LC_t, NE_t, C_t, Pupil_t, LC_raw, NE_raw, C_raw
        return LC_t, NE_t, C_t, Pupil_t

class LCNECortexLSTM(nn.Module):
    """LCNECortex model with LSTM-style gating mechanisms for improved memory dynamics"""
    def __init__(self, input_dim, hidden_dim=8, lambda_cortex=0.1):
        super(LCNECortexLSTM, self).__init__()
        self.lambda_cortex = lambda_cortex
        self.hidden_dim = hidden_dim

        # Linear layers for core operations
        self.W_x = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC = nn.Linear(hidden_dim, hidden_dim)
        self.W_C = nn.Linear(hidden_dim, hidden_dim)
        self.W_Pupil = nn.Linear(hidden_dim, 1)  

        # LSTM-like gating mechanisms
        self.W_forget = nn.Linear(hidden_dim, hidden_dim)   # Forget gate
        self.W_input = nn.Linear(hidden_dim, hidden_dim)    # Input gate
        self.W_output = nn.Linear(hidden_dim, hidden_dim)   # Output gate
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.relu = nn.LeakyReLU(0.1)

    def forward(self, X, prev_LC, prev_Cortex, cell_state, return_activations=False):
        x_input = self.W_x(X)

        # LC activity
        LC_raw = x_input + self.W_h(prev_LC)  
        LC_t = self.relu(self.bn1(LC_raw))

        # NE release
        NE_raw = self.W_LC(LC_t)
        NE_t = self.relu(self.bn2(NE_raw))

        # Forget & Input gates
        forget_gate = torch.sigmoid(self.W_forget(prev_Cortex))  # Forget previous memory
        input_gate = torch.sigmoid(self.W_input(x_input))  # Determine new memory contribution
        
        # Update Cortex Memory (Cell State)
        cell_state = forget_gate * cell_state + input_gate * self.lambda_cortex * self.W_LC(NE_t) + self.W_C(x_input)
        C_t = self.relu(self.bn3(cell_state))  # Cortex activation

        # Output gate
        output_gate = torch.sigmoid(self.W_output(C_t))
        Pupil_t = output_gate * self.W_Pupil(LC_t + C_t + NE_t)  # Modulated pupil dilation

        if return_activations:
            return LC_t, NE_t, C_t, Pupil_t, forget_gate, input_gate, output_gate, cell_state
        return LC_t, NE_t, C_t, Pupil_t, cell_state

