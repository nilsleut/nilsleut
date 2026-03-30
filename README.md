# Nils Leutenegger
Gymnasium student (Zentralschweiz, Switzerland).  
Interested in NeuroAI and the mathematics of learning — how neural networks represent information, and how that relates to biological systems.

## Projects

### [Predictive Coding and the Visual Cortex](https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex)
Tested whether a Predictive Coding network — a model built according to the brain's own computational logic — represents visual information the way the human visual cortex does. Trained a hierarchical PC network on ResNet-50 features and compared its internal representations to 7T fMRI data (THINGS-fMRI, N=3 subjects) using Representational Similarity Analysis across six cortical ROIs (V1–IT). Key finding: PC layers show a crossing gradient across the visual hierarchy (interaction effect Δr₀−Δr₃ = +0.266, replicated in all 3 subjects), and PC r₀ outperforms ViT-B/16 and CLIP at V1 — despite those models being trained on orders of magnitude more data.  
`PyTorch` `fMRI` `RSA` `Predictive Coding` `NeuroAI`

### [Predictive Coding vs. Spiking Neural Networks](https://github.com/nilsleut/Predictive-Coding-vs.-Spiking-Neural-Networks)
RSA comparing Predictive Coding (Rao & Ballard 1999) and Spiking Neural Networks (snnTorch) against THINGS-fMRI (N=3 subjects, V1–IT). Key finding: PC develops cortical hierarchy gradient (Δr₀−Δr₃ = +0.266, p=0.007); SNN reaches 91% PC-IT performance but lacks hierarchy. PC r₃ beats ResNet-50 at LOC/IT.  
`PyTorch` `snnTorch` `fMRI` `RSA` `Predictive Coding` `Spiking NNs` `NeuroAI`

### [nanoGPT Ablation: Depth & Head Scaling](https://github.com/nilsleut/nanogpt-ablation-fineweb)
Systematic ablation study on transformer scaling using nanoGPT on FineWeb-Edu (~104M tokens). Isolated depth (n_layer: 2,4,6,8 at n_head=6) and heads (n_head: 1,2,3 at n_layer=6, d=384). Key finding: 6L-3H optimal (val loss 5.32, PPL 204); head scaling saturates at ~128 dim/head (+1.5% loss reduction per doubling), depth shows quadratic diminishing returns beyond 6 layers.  
`PyTorch` `nanoGPT` `Transformers` `Ablation Study`

### [RSA — Comparing CNNs and Transformers to Human Visual Cortex](./rsa-things-fmri)
Representational Similarity Analysis on the THINGS-fMRI dataset. Compared ResNet-50, ViT-B/16, and CLIP across six visual areas (V1–IT). Key finding: visual hierarchy explains IT cortex representations better than language-grounded semantics — a negative result for CLIP.  
`PyTorch` `fMRI` `RSA` `NeuroAI`

### [Grokking — Delayed Generalization in Modular Arithmetic](./grokking)
Reproduction of Power et al. (2022). 1-layer Transformer trained on (a+b) mod 97. Includes phase diagram mapping the transition between direct generalization, grokking, and failure to generalize across dataset size and weight decay.  
`PyTorch` `Transformers` `Generalization`

### [DQN — Atari Breakout](./dqn-breakout)
Paper-faithful implementation of Mnih et al. (2015). Replay buffer, target network, frame stacking, reward clipping. Trained for 5M frames.  
`PyTorch` `Reinforcement Learning` `Atari`

### [MLP from Scratch — Maturaarbeit](./mlp-maturaarbeit)
Fully-connected neural network implemented in NumPy without frameworks. Backpropagation derived and coded manually. Evaluated on MNIST.  
`NumPy` `Backpropagation` `From Scratch`

## Background
- Swiss Physics Olympiad, Round 2
- Swiss Informatic Olympiad, Round 2
- Read: Goodfellow et al. *Deep Learning*; Prince *Understanding Deep Learning*; Gerstner et al. *Neuronal Dynamics*; Dayan & Abbott *Theoretical Neuroscience*

## Contact
Open to research internship opportunities in NeuroAI or ML theory starting autumn 2026.
