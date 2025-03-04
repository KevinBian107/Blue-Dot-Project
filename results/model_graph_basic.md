The following is our basic `feed forward gadget model with uncertainty reasoning`:

```mermaid
graph TD;
    %% FF Gadget Controller
    A["Input x"] --> B["Hidden 1 (ReLU)"];
    B --> C["Hidden 2 (ReLU)"];
    
    C --> LC_Input["Pass to LCNEGadget"];
    
    %% LCNEGadget System
    LC_Input --> LC1["W_LC1 (Tanh) → Intermediate LC"];
    LC1 --> LC2["$W_LC2$ (Tanh) → LC_t"];
    
    LC2 --> Tonic["Tonic Control (Tanh) * Gain → Tonic NE"];
    LC2 --> Phasic["Phasic Control (Tanh) * Gain → Phasic NE"];
    Phasic --> Suppression["Suppression Factor (Sigmoid) * Phasic NE"];

    Tonic --> Blend;
    Suppression --> Blend;
    Gate --> Blend["Blending: Tonic NE + Phasic NE"];
    Blend --> NE_t["Final NE_t"];
    
    %% Connection back to FF Gadget Controller
    NE_t --> H["Concat with Hidden 2"];
    C --> H;
    
    H --> Modulation["modulation_fc → Modulated Hidden (ReLU)"];
    Modulation --> Mean["Pupil_mean (Linear) → Pupil Dilation Prediction"];
    Modulation --> Var["Pupil_var (Exp(Linear)) → Pupil Dilation Uncertainty"];

    Mean --> Output["Final Pupil Dilation Output"];
    Var --> Output;