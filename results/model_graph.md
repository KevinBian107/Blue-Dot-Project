The following is our `feed forward gadget model with uncertainty reasoning`:

```mermaid
graph LR;
    
    %% Inputs
    A["Input x"] -->|ReLU| B["NN Hidden State"];
    
    subgraph LCNEGadget["LCNE Gadget"]
        %% LC-NE Gadget Processing
        B --> |tanh| LC["LC Hidden State"];

        %% Integrating Past LC Activations
        subgraph MemoryIntegration["Past LC Integration"]
            PrevLC["Previous LC_{t-1}"] -->|Adaptive Weighting| WeightedPast["Weighted Past LC"];
            WeightedPast -->|Sigmoid| AddPast["LC_t + Weighted Past"];
        end
        LC --> AddPast;

        %% Neuromodulation
        AddPast -->|Tanh * Gain| Tonic["Tonic NE"];
        AddPast -->|Tanh * Gain| Phasic["Phasic NE"];
        Tonic --> NE_t
        Phasic --> NE_t

        %% Uncertainty Modulation
        AddPast -->|Sigmoid| UncertaintyGate["Uncertainty Gate"];
        UncertaintyGate -->|Scaling| NE_t["Final NE_t"];
    end
    
    %% FF Gadget Controller Integration
    NE_t --> H["Skip Connect Hidden State"];
    B --> H;
    
    H -->|ReLU| Modulation["modulation_fc → Modulated Hidden"];
    Modulation --> Mean["Pupil_mean (Linear) → Pupil Dilation Prediction"];
    Modulation --> Var["Pupil_var (Exp(Linear)) → Pupil Dilation Uncertainty"];

    Mean --> Output["Final Pupil Dilation"];
    Var --> Output;
