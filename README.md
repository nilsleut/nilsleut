# Nils Leutenegger

Pre-university researcher (Zentralschweiz, Switzerland) 
Working on NeuroAI — how biological learning rules shape neural representations, and what that means for machine learning.

📄 **Preprint:** [Untrained CNNs Match Backpropagation at V1](https://arxiv.org/abs/2604.16875) — arXiv 2026
---

## Projects

### [Learning Rules RSA: BP, FA, PC, STDP vs. Human fMRI](https://github.com/nilsleut/learning-rules-rsa)
Systematic RSA comparison of four learning rules (Backpropagation, Feedback Alignment, Predictive Coding, STDP) plus an untrained baseline against THINGS-fMRI (720 stimuli, N=3 subjects, V1–IT). Key findings: V1 alignment is architecture-driven — an untrained CNN matches BP (p=0.43); learning rules only differentiate at LOC/IT where BP dominates (d>2.3); PC matches BP at IT (p=0.18) using only local Hebbian updates; FA actively degrades representations below the random baseline (d=1.1). Includes permutation tests (N=1000), bootstrap CIs, partial RSA, and filter analysis.  
`PyTorch` `fMRI` `RSA` `Predictive Coding` `STDP` `Feedback Alignment` `NeuroAI`

### [Biologically Plausible RL Plays Pong](https://github.com/nilsleut/bio-plausible-rl-pong)
Can a fully backprop-free agent learn Pong? Custom environment, PPO from scratch as baseline, then a biologically plausible agent combining Predictive Coding for feature learning with distributional Hebbian plasticity for value estimation (inspired by Dabney et al. 2020). The Hebbian agent matches PPO (61% vs 59%) with engineered features; the full BioAgent (PC + Hebbian, zero backprop) reaches 57%. Self-play training works in principle but exposes the plasticity-stability dilemma. Everything from scratch, ~1500 lines.  
`PyTorch` `Reinforcement Learning` `PPO` `Predictive Coding` `Hebbian Learning` `Self-Play` `From Scratch`

### [Predictive Coding Inference Dynamics vs. Learning Rules in RSA](https://github.com/nilsleut/Predictive-Coding-Inference-Dynamics-vs.-Learning-Rules-in-RSA)
Extension of *Predictive Coding and the Visual Cortex* testing whether the cortical hierarchy gradient in RSA against 7T fMRI is produced by the weight update rule or by the inference dynamics of the PC network. Null result: the gradient is driven by inference dynamics and ResNet initialisation, not the learning rule — honestly framed.  
`PyTorch` `fMRI` `RSA` `Predictive Coding` `NeuroAI`

### [Predictive Coding vs. Spiking Neural Networks](https://github.com/nilsleut/Predictive-Coding-vs.-Spiking-Neural-Networks)
RSA comparing Predictive Coding (Rao & Ballard 1999) and Spiking Neural Networks (snnTorch) against THINGS-fMRI (N=3 subjects, V1–IT). Key finding: PC develops a cortical hierarchy gradient (Δr₀−Δr₃ = +0.266, p=0.007, replicated in all 3 subjects); SNN reaches 92% of PC-IT performance but lacks the cross-over.  
`PyTorch` `snnTorch` `fMRI` `RSA` `Predictive Coding` `NeuroAI`

### [Predictive Coding and the Visual Cortex](https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex)
Hierarchical PC network trained on ResNet-50 features, compared to 7T fMRI (THINGS-fMRI, N=3 subjects) via RSA across six cortical ROIs (V1–IT). PC layers show a crossing gradient: r₀ correlates most strongly with V1 (ρ=0.30), r₃ with IT (ρ=0.16).  
`PyTorch` `fMRI` `RSA` `Predictive Coding` `NeuroAI`

### [Grokking — Fourier Features & Mechanistic Interpretability](https://github.com/nilsleut/grokking-mechanistic-interpretability)
Replication and extension of Nanda et al. (2023) on modular arithmetic (a+b) mod p. Four lines of evidence that the trained model implements a Fourier multiplication algorithm. Multi-modulus analysis across p ∈ {71, 83, 97, 113}: frequencies are within-p stable across seeds but do not cluster at universal harmonic ratios.  
`PyTorch` `Mechanistic Interpretability` `Transformers` `Fourier Analysis`

### [EventSNN — Gesture Recognition on DVS Event Camera Data](https://github.com/nilsleut/EventSNN-Gesture-Recognition-on-DVS-Event-Camera-Data)
LIF spiking neural network for gesture classification on the IBM DVS128 dataset (11 classes). DVS cameras fire asynchronous events like retinal ganglion cells — pairing them with an SNN creates a doubly bio-inspired pipeline. 67% accuracy (chance: 9.1%).  
`PyTorch` `snnTorch` `Event Camera` `Neuromorphic` `NeuroAI`

### [nanoGPT Ablation: Depth and Head Scaling](https://github.com/nilsleut/nanoGPT-Ablation-Study)
Systematic ablation on transformer scaling using nanoGPT on FineWeb-Edu (~104M tokens). Depth has a strong, monotonically decreasing effect on val loss; head count has almost no effect at fixed depth.  
`PyTorch` `nanoGPT` `Transformers` `Ablation Study`

### [RSA — CNNs and Transformers vs. Human Visual Cortex](https://github.com/nilsleut/Representational-Similarity-Analysis-THINGS-fMRI)
RSA on THINGS-fMRI comparing ResNet-50, ViT-B/16, and CLIP across V1–IT. Visual hierarchy explains IT representations better than language-grounded semantics — a negative result for CLIP.  
`PyTorch` `fMRI` `RSA` `NeuroAI`

### [Monte Carlo Options Pricing](https://github.com/nilsleut/Monte-Carlo-Options-Pricing)
European options with variance reduction; American calls via Longstaff-Schwartz LSMC.  
`NumPy` `SciPy` `Stochastic Processes` `Quantitative Finance`

### [Grokking — Delayed Generalization](https://github.com/nilsleut/Grokking-Generalization-Beyond-Overfitting)
Reproduction of Power et al. (2022). Phase diagram mapping memorisation, grokking, and failure across dataset size and weight decay.  
`PyTorch` `Transformers` `Generalization`

### [DQN — Atari Breakout](https://github.com/nilsleut/Deep-Q-Network-Atari-Breakout)
Paper-faithful Mnih et al. (2015). Replay buffer, target network, frame stacking. 5M frames, ~17 mean reward.  
`PyTorch` `Reinforcement Learning` `Atari`

### [MLP from Scratch — Maturaarbeit](https://github.com/nilsleut/Maturaarbeit_NilsLeutenegger)
Neural network in NumPy, backprop derived manually. Swiss referendum turnout prediction. Grade: 6/6.  
`NumPy` `Backpropagation` `From Scratch`

---

## Background

- Swiss Physics Olympiad, Round 2  
- Swiss Informatics Olympiad, Round 2  
- Read: Goodfellow et al. *Deep Learning*; Prince *Understanding Deep Learning*; Gerstner et al. *Neuronal Dynamics*; Dayan & Abbott *Theoretical Neuroscience*

---

## Contact

Open to research internship opportunities in NeuroAI or ML theory starting autumn 2026.
