import torch
import torch.nn as nn

class LCNEGadget(nn.Module):
    """Self-learned LC-NE system with better Phasic NE adaptation."""
    
    def __init__(self, hidden_dim):
        super(LCNEGadget, self).__init__()
        self.hidden_dim = hidden_dim

        # LC transformation for more expressivity
        self.W_LC1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC2 = nn.Linear(hidden_dim, hidden_dim)

        self.tonic_control = nn.Linear(hidden_dim, hidden_dim)
        self.phasic_control = nn.Linear(hidden_dim, hidden_dim)

        # learnable NE scaling
        self.tonic_gain = nn.Parameter(torch.ones(hidden_dim))  
        self.phasic_gain = nn.Parameter(torch.ones(hidden_dim))  

        # adaptive suppression factor for Phasic NE
        self.suppression_factor = nn.Parameter(torch.ones(hidden_dim))

        # gating function learns how to balance tonic & phasic NE
        self.gate = nn.Linear(hidden_dim, hidden_dim)

        # normalization layer (prevents runaway NE values)
        self.norm = nn.LayerNorm(hidden_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_state):
        """Infers NE modulation dynamically and normalizes output."""
        
        LC_t = self.tanh(self.W_LC1(hidden_state))
        LC_t = self.tanh(self.W_LC2(LC_t))

        tonic_NE = self.tanh(self.tonic_control(LC_t)) * self.tonic_gain  
        phasic_NE = self.tanh(self.phasic_control(LC_t)) * self.phasic_gain  
        phasic_NE *= self.sigmoid(self.suppression_factor * LC_t)

        gating_factor = self.sigmoid(self.gate(LC_t))  
        NE_t = (1 - gating_factor) * tonic_NE + gating_factor * phasic_NE  
        
        NE_t = self.norm(NE_t)  

        return LC_t, NE_t, tonic_NE, phasic_NE

class FFGadgetController(nn.Module):
    """FF network learns to control LC-NE system for modulation"""
    
    def __init__(self, input_dim, hidden_dim):
        super(FFGadgetController, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.lcne_gadget = LCNEGadget(hidden_dim) 

        self.modulation_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # tonic & phasic NE
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, activation=False):
        """Processes input and learns to utilize neuromodulation"""
        hidden_1 = torch.relu(self.fc1(x))
        hidden_2 = torch.relu(self.fc2(hidden_1))

        # neuromodulatory signals
        LC_t, NE_t, tonic_NE, phasic_NE = self.lcne_gadget(hidden_2)

        # FFN Learns How to Use Neuromodulation through residuals
        modulated_input = torch.cat([hidden_2, NE_t], dim=1)
        modulated_hidden = torch.relu(self.modulation_fc(modulated_input))
        output = self.output_layer(modulated_hidden)

        if activation:
            return output, LC_t, NE_t, tonic_NE, phasic_NE, hidden_1, hidden_2

        return output, LC_t, NE_t, tonic_NE, phasic_NE


class LCNEGadgetCompositional(nn.Module):
    """LC-NE system that integrates past activations implicitly for adaptive modulation.
    Incoportaes self-regulation of the gadget through damping and amplifying the effect"""
    
    def __init__(self, hidden_dim):
        super(LCNEGadgetCompositional, self).__init__()
        self.hidden_dim = hidden_dim

        # LC transformation layers for hierarchical processing
        self.W_LC1 = nn.Linear(hidden_dim, hidden_dim)
        self.W_LC2 = nn.Linear(hidden_dim, hidden_dim)

        self.tonic_control = nn.Linear(hidden_dim, hidden_dim)
        self.phasic_control = nn.Linear(hidden_dim, hidden_dim)

        # adaptive weighting for historical LC activations
        self.adaptive_weight = nn.Linear(hidden_dim, hidden_dim)  

        # dynamic uncertainty regulation
        self.uncertainty_gate = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # store past LC states
        self.prev_LC = None  

    def forward(self, hidden_state):
        """Processes LC activations dynamically, integrating past activations while handling batch size changes."""
        
        # Tanh is Centered Around Zero → Balanced Excitation/Inhibition
        LC_t = self.tanh(self.W_LC1(hidden_state))
        LC_t = self.tanh(self.W_LC2(LC_t))

        if self.prev_LC is not None:
            
            if self.prev_LC.shape[0] != LC_t.shape[0]:
                print(f"Adjusting prev_LC size: {self.prev_LC.shape} -> {LC_t.shape}")
                self.prev_LC = LC_t.detach().clone() 
            
            weighted_past = self.sigmoid(self.adaptive_weight(self.prev_LC)) * self.prev_LC
            LC_t = LC_t + weighted_past 
        
        self.prev_LC = LC_t.detach().clone()

        # neuromodulatory signals
        tonic_NE = self.tanh(self.tonic_control(LC_t))  
        phasic_NE = self.tanh(self.phasic_control(LC_t))  

        # adaptive uncertainty gating
        uncertainty_weight = self.sigmoid(self.uncertainty_gate(LC_t))
        NE_t = (1 - uncertainty_weight) * tonic_NE + uncertainty_weight * phasic_NE

        NE_t = self.norm(NE_t)  

        return LC_t, NE_t, tonic_NE, phasic_NE


class FFGadgetUncertainController(nn.Module):
    """FF network learns to control LC-NE system for pupil dilation + uncertainty."""
    
    def __init__(self, input_dim, hidden_dim):
        super(FFGadgetUncertainController, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.lcne_gadget = LCNEGadgetCompositional(hidden_dim) 

        self.modulation_fc = nn.Linear(hidden_dim * 2, hidden_dim)  # tonic & phasic NE
        
        self.uncertainty_scale = nn.Parameter(torch.tensor(1.0))  

        # Pupil Dilation Output + Uncertainty
        self.Pupil_mean = nn.Linear(hidden_dim, 1)  # Pupil Dilation Prediction
        self.Pupil_var = nn.Linear(hidden_dim, 1)   # Uncertainty for Pupil Dilation (std dev)

    def forward(self, x, activation=False):
        """Processes input and learns to utilize neuromodulation"""
        hidden_1 = torch.relu(self.fc1(x))
        hidden_2 = torch.relu(self.fc2(hidden_1))

        # neuromodulatory signals controled by the nerual network, a self-regulating gadget that it need to learn to play with
        LC_t, NE_t, tonic_NE, phasic_NE = self.lcne_gadget(hidden_2)

        # FFN Learns How to Use Neuromodulation
        modulated_input = torch.cat([hidden_2, NE_t], dim=1)
        modulated_hidden = torch.relu(self.modulation_fc(modulated_input))

        pupil_mean = self.Pupil_mean(modulated_hidden)
        # pupil_var = torch.exp(torch.clamp(self.Pupil_var(modulated_hidden) + torch.randn_like(self.Pupil_var(modulated_hidden)) * 0.1, min=-5, max=2))
        pupil_var = torch.exp(self.Pupil_var(modulated_hidden) * self.uncertainty_scale)

        if activation:
            return pupil_mean, pupil_var, LC_t, NE_t, tonic_NE, phasic_NE, hidden_1, hidden_2

        return pupil_mean, pupil_var