# Blue Dot Project: A Neural Gadget Controller
Thi is a study into Locus Coeruleus Norepinephrine System and Security Motivation System rumination pathology pathway using computational methods. To this end, we are creating a Locus Coeruleus rumination system in which we call it as the **B**lue **D**ot (LC in greek means little blue dot) **P**roject (BDP).

- [Meeting docs](https://docs.google.com/document/d/1740GxJ5xmIjUbWH8_RjYnuI5KNnZwGkvKQaM6hEzCLc/edit?tab=t.0#heading=h.e9mhf81r5r4b)
- [Pitch presentation model proposition section (hosted in UCSD Podemos)](https://www.youtube.com/watch?v=KdRzyWItoa0)

## Neural Gadget Controller Models:
We try to leverage neurosciecne behavior data related to pupil dilation & memory to build a data driven model. We want to fit and manipulate the model component to give us a platform to operate on and play around with it. We want to  achieve 3 goals:

1. we want to use the the idea of building an **mechanistic model**:
    - Very interpretable model as we control each "mechanistic" controller in seeing what might happen if we change something.

2. we want to create a **gadget** for network to use:
    - Instead of modifying the models directly, maybe we can take an approach that the Neural Turing Machine (NTM) did and try to create a gadget (in NTM it represents memory for read and write) and for us the gadget can be think of as teh LC-NE system.
    - It need to be differentiable so everything can be back propoagted and we can see if the network can learn to use this controller (LC-NE system) to achieve the effect we see in the samples.

3. We  want to represent abstract concepts in **internal state** of the model:
    - Conduct persistent homology analysis to see formations of connected components and cyclic cycles as an analogy of thoughts and rumination.
    - Examine uncertainty changes in decision.

We can use this fitteed model to prompt them under certain `experimental` conditions and see hwo they would react to as compares to rea behavior data from animals or human (i.e. the two lick test for rodent).

## Preliminary Results & Analysis:
We provide an [sample notebook](https://github.com/KevinBian107/Blue-Dot-Project/blob/main/simulate.ipynb) for demostration purposes. Now  we have a sand-box that has been setup  to implement all kinds of cool neuroscience ideas that we can do. We have also created [model structure](https://github.com/KevinBian107/Blue-Dot-Project/blob/main/results/model_graph.md) for showcase our pipeline.

| FF Gadget Model Activation Persistent Homology Analysis | FF Gadget Model PCA Activation Network Graph|
|--------------|--------------|
| ![img](results/gadget_homology.png)| ![img](results/gadget_network.png)|


- **𝐻₀ (Blue Dots)** → Represents connected components (clusters in LC activation).
- **𝐻₁ (Orange Dots)** → Represents loops (1D holes in the data, cyclic structures in LC activation).
- **Diagonal Line** → Features close to the diagonal are short-lived (noise).
- **Dashed Line (∞)** → Features that never die represent persistent structures in the data.

> 1️⃣ **𝐻₀ (Blue Dots) → Connected Components (Stable Thought States)**
- Represents distinct activation states of LC.
- Few long-lived components → Suggests stable attractor states in LC activation, possibly corresponding to persistent thought patterns.
- Many short-lived components → Indicates a highly dynamic LC state, suggesting flexible thought processes rather than being stuck in loops.

> **Key**: connected components indicates thoughts.

Rumination:
- If **𝐻₀ features persist (far from diagonal)**, this could indicate LC stabilization, linked to difficulty in shifting thoughts (rumination).
- If **𝐻₀ features quickly disappear**, LC activity may dynamically support cognitive flexibility, allowing for thought transitions.

> 2️⃣ **𝐻₁ (Orange Dots) → Cycles in LC Activity (Recursive Thoughts)**
- Represents loops in activation patterns, indicating recurrent or cyclic thought processes.
- Few short-lived loops → Suggests transient, non-repetitive activity.
- Many persistent loops → Could indicate self-reinforcing thought cycles, similar to recursive thinking in rumination.

Rumination:
- Persistent loops (far from diagonal) may correspond to sustained LC activity patterns, reinforcing self-referential thought loops seen in rumination.
- Short-lived loops (near diagonal) suggest transient fluctuations in LC activity, possibly linked to shifts in attention or task engagement.

> **Key**: connected components indicates close/open loops of thoughts.

## Data Source:
The [openneuro](https://openneuro.org/) is a very good source of data, we are specifically using this ["Locus coeruleus activity strengthens prioritized memories under arousal"](https://openneuro.org/datasets/ds002011/versions/1.0.0) dataset for now.