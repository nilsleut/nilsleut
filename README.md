# Nils Leutenegger

Pre-university student (Zentralschweiz, Switzerland), targeting ETH Informatik.  
Interested in NeuroAI and the mathematics of learning — how neural networks represent information, and how that relates to biological systems.

---

## Projects

### [Learning Rules RSA: BP, FA, PC, STDP vs. Human fMRI](https://github.com/nilsleut/learning-rules-rsa)
Systematic RSA comparison of four learning rules (Backpropagation, Feedback Alignment, Predictive Coding, STDP) plus an untrained baseline against THINGS-fMRI (720 stimuli, N=3 subjects, V1–IT). Key findings: V1 alignment is architecture-driven — an untrained CNN matches BP (p=0.43); learning rules only differentiate at LOC/IT where BP dominates (d>2.3); PC matches BP at IT (p=0.18) using only local Hebbian updates; FA actively degrades representations below the random baseline (d=1.1). Includes permutation tests (N=1000), bootstrap CIs, partial RSA, and filter analysis.  
`PyTorch` `fMRI` `RSA` `Predictive Coding` `STDP` `Feedback Alignment` `NeuroAI`

### [Predictive Coding Inference Dynamics vs. Learning Rules in RSA](https://github.com/nilsleut/Predictive-Coding-Inference-Dynamics-vs.-Learning-Rules-in-RSA)
Extension of *Predictive Coding and the Visual Cortex* testing whether the cortical hierarchy gradient in RSA against 7T fMRI is produced by the weight update rule or by the inference dynamics of the PC network. Null result: the gradient is driven by inference dynamics and ResNet initialisation, not the learning rule — honestly framed.  
`PyTorch` `fMRI` `RSA` `Predictive Coding` `NeuroAI`

### [Predictive Coding vs. Spiking Neural Networks](https://github.com/nilsleut/Predictive-Coding-vs.-Spiking-Neural-Networks)
RSA comparing Predictive Coding (Rao & Ballard 1999) and Spiking Neural Networks (snnTorch) against THINGS-fMRI (N=3 subjects, V1–IT). Key finding: PC develops a cortical hierarchy gradient (Δr₀−Δr₃ = +0.266, p=0.007, replicated in all 3 subjects); SNN reaches 92% of PC-IT performance but lacks the cross-over. PC r₃ outperforms ResNet-50 at LOC and IT.  
`PyTorch` `snnTorch` `fMRI` `RSA` `Predictive Coding` `NeuroAI`

### [Predictive Coding and the Visual Cortex](https://github.com/nilsleut/Predictive-Coding-and-the-Visual-Cortex)
Hierarchical PC network trained on ResNet-50 features, compared to 7T fMRI (THINGS-fMRI, N=3 subjects) via RSA across six cortical ROIs (V1–IT). PC layers show a crossing gradient: r₀ correlates most strongly with V1 (ρ=0.30), r₃ with IT (ρ=0.16). PC r₀ outperforms ViT-B/16 and CLIP at V1 despite those models being trained on orders of magnitude more data.  
`PyTorch` `fMRI` `RSA` `Predictive Coding` `NeuroAI`

### [Grokking — Fourier Features & Mechanistic Interpretability](https://github.com/nilsleut/grokking-mechanistic-interpretability)
Replication and extension of Nanda et al. (2023) on modular arithmetic (a+b) mod p. Part 1: four lines of evidence that the trained model implements a Fourier multiplication algorithm — sparse W_E spectrum (Gini=0.605), W_L effective rank ~10 (95.9% energy in top-10 SVs), 2D-FFT grid structure, and ablation (keys removed → 1.7% accuracy). Part 2: multi-modulus analysis across p ∈ {71, 83, 97, 113} to test whether key frequencies are structurally bound to p. Finding: frequencies are within-p stable across seeds but do not cluster at universal harmonic ratios — frequency selection is p-specific.  
`PyTorch` `Mechanistic Interpretability` `Transformers` `Fourier Analysis`

### [EventSNN — Gesture Recognition on DVS Event Camera Data](https://github.com/nilsleut/EventSNN-Gesture-Recognition-on-DVS-Event-Camera-Data)
LIF spiking neural network for gesture classification on the IBM DVS128 dataset (11 classes). DVS cameras fire asynchronous events like retinal ganglion cells — pairing them with an SNN creates a doubly bio-inspired pipeline. Achieves 67% validation accuracy with a fully-connected architecture (chance: 9.1%). Documents known limitations: FC-only ignores spatial structure, rate coding discards spike timing precision.  
`PyTorch` `snnTorch` `Event Camera` `Neuromorphic` `NeuroAI`

### [nanoGPT Ablation: Depth and Head Scaling](https://github.com/nilsleut/nanoGPT-Ablation-Study)
Systematic ablation on transformer scaling using nanoGPT on FineWeb-Edu (~104M tokens, 5000 iterations per run). Key findings: depth has a strong, monotonically decreasing effect on val loss (L2→L4: −0.160, L6→L8: −0.033); head count has almost no effect — 1 head vs 6 heads differs by only 0.029 val loss at fixed depth. Depth dominates at small scale.  
`PyTorch` `nanoGPT` `Transformers` `Ablation Study`

### [RSA — Comparing CNNs and Transformers to Human Visual Cortex](https://github.com/nilsleut/Representational-Similarity-Analysis-THINGS-fMRI)
Representational Similarity Analysis on THINGS-fMRI. Compared ResNet-50, ViT-B/16, and CLIP across six visual areas (V1–IT). Key finding: visual hierarchy explains IT cortex representations better than language-grounded semantics — a negative result for CLIP.  
`PyTorch` `fMRI` `RSA` `NeuroAI`

### [Monte Carlo Options Pricing](https://github.com/nilsleut/Monte-Carlo-Options-Pricing)
Implementation of Monte Carlo methods for pricing European and American options under the Black-Scholes model. European options with variance reduction techniques; American calls via Longstaff-Schwartz (2001) LSMC algorithm; call/put comparison confirming Merton (1973).  
`NumPy` `SciPy` `Stochastic Processes` `Quantitative Finance`

### [Grokking — Delayed Generalization in Modular Arithmetic](https://github.com/nilsleut/Grokking-Generalization-Beyond-Overfitting)
Reproduction of Power et al. (2022). 1-layer Transformer trained on (a+b) mod 97. Includes phase diagram mapping the transition between memorisation, grokking, and failure to generalise across dataset size and weight decay.  
`PyTorch` `Transformers` `Generalization`

### [DQN — Atari Breakout](https://github.com/nilsleut/Deep-Q-Network-Atari-Breakout)
Paper-faithful implementation of Mnih et al. (2015). Replay buffer, target network, frame stacking, reward clipping. Trained for 5M frames, achieving ~17 mean reward.  
`PyTorch` `Reinforcement Learning` `Atari`

### [MLP from Scratch — Maturaarbeit](https://github.com/nilsleut/Maturaarbeit_NilsLeutenegger)
Fully-connected neural network implemented in NumPy without frameworks. Backpropagation derived and coded manually. Applied to Swiss referendum turnout prediction. Grade: 6/6.  
`NumPy` `Backpropagation` `From Scratch`

---

## Background

- Swiss Physics Olympiad, Round 2
- Swiss Informatics Olympiad, Round 2
- Read: Goodfellow et al. *Deep Learning*; Prince *Understanding Deep Learning*; Gerstner et al. *Neuronal Dynamics*; Dayan & Abbott *Theoretical Neuroscience*

---

## Contact

Open to research internship opportunities in NeuroAI or ML theory starting autumn 2026.
