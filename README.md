# Nils Leutenegger

Pre-university NeuroAI researcher · Zentralschweiz, Switzerland  

---

**Research interest:** How biological learning rules shape neural representations — and what that tells us about machine learning.

📄 **Preprint (arXiv 2026):** [Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI](https://arxiv.org/abs/2604.16875)

---

## NeuroAI

### [Learning Rules RSA: BP, FA, PC, STDP vs. Human fMRI](https://github.com/nilsleut/learning-rules-rsa)
Systematic RSA comparison of four learning rules against THINGS-fMRI (720 stimuli, N=3 subjects, V1–IT). Permutation tests (N=1000), bootstrap CIs, partial RSA, filter analysis.  
**Key findings:** V1 alignment is architecture-driven — an untrained CNN matches BP (p=0.43). Learning rules only differentiate at LOC/IT: BP dominates (d>2.3); PC matches BP at IT (p=0.18) using only local Hebbian updates; FA actively degrades representations below the random baseline (d=1.1).  
`PyTorch` `RSA` `fMRI` `Predictive Coding` `STDP` `Feedback Alignment`

### [Biologically Plausible RL Plays Pong](https://github.com/nilsleut/Biologically-Plausible-RL-Plays-Pong)
Can a fully backprop-free agent learn Pong? PPO baseline from scratch, then a BioAgent combining Predictive Coding for feature learning with distributional Hebbian plasticity for value estimation (Dabney et al. 2020). Everything from scratch, ~1500 lines.  
**Key findings:** BioAgent (PC + Hebbian, zero backprop) reaches 57% vs. PPO's 59%. Self-play training works in principle but exposes the plasticity–stability dilemma.  
`PyTorch` `RL` `PPO` `Predictive Coding` `Hebbian Learning` `Self-Play`

### [Dendritic Encoding](https://github.com/nilsleut/Dendritic-Encoding)
Testing whether a biologically plausible CNN with dendritic compartments produces representations more aligned with the human visual cortex than standard learning rules, evaluated via voxelwise encoding models against fMRI.  
`PyTorch` `fMRI` `CNN` `Dendritic Computation` `NeuroAI`

### [Predictive Coding vs. Spiking Neural Networks](https://github.com/nilsleut/Predictive-Coding-vs.-Spiking-Neural-Networks)
RSA comparing PC (Rao & Ballard 1999) and SNN (snnTorch) against THINGS-fMRI (N=3, V1–IT).  
**Key finding:** PC develops a cortical hierarchy gradient (Δr₀–Δr₃ = +0.266, p=0.007, replicated in all 3 subjects); SNN reaches 92% of PC-IT performance but lacks the crossover.  
`PyTorch` `snnTorch` `RSA` `fMRI` `Predictive Coding`

### [Predictive Coding Inference Dynamics vs. Learning Rules](https://github.com/nilsleut/Predictive-Coding-Inference-Dynamics-vs.-Learning-Rules-in-RSA)
Tests whether the cortical hierarchy gradient in RSA is driven by the weight update rule or by PC inference dynamics. Null result: the gradient is driven by inference dynamics and ResNet initialisation — honestly framed.  
`PyTorch` `RSA` `fMRI` `Predictive Coding`

### [Predictive Coding and the Visual Cortex](https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex)
Hierarchical PC network compared to 7T fMRI (THINGS-fMRI, N=3) via RSA across six cortical ROIs (V1–IT). PC layers show a crossing gradient: r₀ correlates most strongly with V1 (ρ=0.30), r₃ with IT (ρ=0.16).  
`PyTorch` `RSA` `fMRI` `Predictive Coding`

### [EventSNN — Gesture Recognition on DVS Event Camera Data](https://github.com/nilsleut/EventSNN-Gesture-Recognition-on-DVS-Event-Camera-Data)
LIF spiking network for gesture classification on IBM DVS128 (11 classes). DVS cameras fire like retinal ganglion cells — a doubly bio-inspired pipeline. 67% accuracy (chance: 9.1%).  
`PyTorch` `snnTorch` `Neuromorphic` `Event Camera`

### [RSA — CNNs and Transformers vs. Human Visual Cortex](https://github.com/nilsleut/Representational-Similarity-Analysis-THINGS-fMRI)
RSA on THINGS-fMRI comparing ResNet-50, ViT-B/16, and CLIP across V1–IT. Visual hierarchy explains IT representations better than language-grounded semantics — a negative result for CLIP.  
`PyTorch` `RSA` `fMRI`

---

## Interpretability & ML

### [Grokking — Fourier Features & Mechanistic Interpretability](https://github.com/nilsleut/grokking-mechanistic-interpretability)
Replication of Nanda et al. (2023) on modular arithmetic. Four lines of evidence for a Fourier multiplication algorithm. Multi-modulus analysis across p ∈ {71, 83, 97, 113}: frequencies are within-p stable across seeds but do not cluster at universal harmonic ratios.  
`PyTorch` `Mechanistic Interpretability` `Transformers` `Fourier Analysis`

### [nanoGPT Ablation: Depth and Head Scaling](https://github.com/nilsleut/nanoGPT-Ablation-Study)
Systematic ablation on transformer scaling using nanoGPT on FineWeb-Edu (~104M tokens). Depth has a strong, monotonically decreasing effect on val loss; head count has almost no effect at fixed depth.  
`PyTorch` `Transformers` `Ablation Study`

### [Grokking — Delayed Generalization](https://github.com/nilsleut/Grokking-Generalization-Beyond-Overfitting)
Reproduction of Power et al. (2022). Phase diagram mapping memorisation, grokking, and failure across dataset size and weight decay.  
`PyTorch` `Transformers`

### [DQN — Atari Breakout](https://github.com/nilsleut/Deep-Q-Network-Atari-Breakout)
Paper-faithful Mnih et al. (2015). Replay buffer, target network, frame stacking. 5M frames, ~17 mean reward.  
`PyTorch` `RL`

---

## Other

### [Swiss Referendum Predictor — Full-Stack ML System](https://github.com/nilsleut/swiss-referendum-predictor)
End-to-end production ML system: PyTorch MLP trained on 6 083 referendums × 584 features, ONNX export, Fastify (TypeScript) API, PostgreSQL, React + Recharts dashboard. Docker Compose, W&B experiment tracking, GitHub Actions CI (ML / API / frontend).  
`PyTorch` `ONNX` `TypeScript` `Fastify` `React` `PostgreSQL` `Docker` `W&B` `CI/CD`

### [Monte Carlo Options Pricing](https://github.com/nilsleut/Monte-Carlo-Options-Pricing)
European options with variance reduction (Antithetic, Control Variates, Quasi-MC/Sobol). American calls via Longstaff-Schwartz LSMC.  
`NumPy` `SciPy` `Stochastic Processes` `Quantitative Finance`

### [MLP from Scratch — Maturaarbeit (6/6)](https://github.com/nilsleut/Maturaarbeit_NilsLeutenegger)
Neural network in NumPy with backprop derived manually. Swiss referendum turnout prediction. Matura grade: 6/6.  
`NumPy` `Backpropagation` `From Scratch`

---

## Background

- Swiss Physics Olympiad, Round 2
- Swiss Informatics Olympiad, Round 2
- Brain-Score PR #2352 (merged)
- Reading: Goodfellow et al. *Deep Learning* · Gerstner et al. *Neuronal Dynamics* · Dayan & Abbott *Theoretical Neuroscience* · Prince *Understanding Deep Learning*

---

Open to research internship opportunities in NeuroAI or ML theory starting autumn 2026.  
📧 nils.leutenegger@gmail.com
