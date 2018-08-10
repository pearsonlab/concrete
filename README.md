# concrete

TensorFlow implementation of variational inference for Hidden Markov Model(HMM) using the continuous relaxation of categorical latent variables described in both [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712) by Chris J. Maddison, Andriy Mnih, Yee Whye Teh and [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) by Eric Jang, Shixiang Gu, Ben Poole.

## Math
1. Gumbel distribution
2. Concrete random variable / Gumbel-Softmax distribution
    1. definition (reparameterization)
    2. properties (categorical under zero temperature; close to uniform under high temperature; convexity under certain conditions)

## Problem Setup & Training Details
1. Section 3.3 & Appendix C of [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712)
    1. the modified training objective / ELBO
    2. Concrete in log-space
    3. temperature (annealing)
2. Section 2.2 & 3.2 of [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
    1. Straight-Through Gumbel-Softmax (Figure 2)
    2. Score Function / REINFORCE & control variates
3. Extension:
    1. gradient estimates
        1. [REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](https://arxiv.org/abs/1703.07370)
        2. [Backpropagation through the Void: Optimizing control variates for black-box gradient estimation](https://arxiv.org/abs/1711.00123)
        3. [ARM: Augment-REINFORCE-Merge Gradient for Discrete Latent Variable Models](https://arxiv.org/abs/1807.11143)
    2. HHMM
        1. [The Hierarchical Hidden Markov Model: Analysis and Applications](https://link.springer.com/article/10.1023/A:1007469218079)
        2. [Linear-time inference in Hierarchical HMMs](http://papers.nips.cc/paper/2050-linear-time-inference-in-hierarchical-hmms)
        3. [Hierarchical Hidden Markov Models with General State Hierarchy](https://aaai.org/Library/AAAI/2004/aaai04-052.php)
        4. [Infinite Hierarchical Hidden Markov Models](http://proceedings.mlr.press/v5/heller09a)

## Experiments
- Gaussian observations, K = 2, N = 500, fixed temperature & exponential decay
