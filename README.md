# concrete

TensorFlow implementation of variational inference for training Hidden Markov Model(HMM) using the continuous relaxation of latent variables described in both [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712) by Chris J. Maddison, Andriy Mnih, Yee Whye Teh and [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) by Eric Jang, Shixiang Gu, Ben Poole.

## Math
1. Gumbel distribution
2. Concrete random variable / Gumbel-softmax distribution
    1. definition
    2. properties (categorical under zero temperature; close to uniform under high temperature; convexity under certain conditions)

## Problem Setup & Training Details
- Section 3.3 & Appendix C of [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712)
    - the modified training objective
    - Concrete in log-space
    - temperature (annealing)
- Section 2.2 & 3.2 of [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
    - Straight-Through Gumbel-Softmax (Figure 2)
    - Score Function / REINFORCE & control variates
- REBAR

## Experiments
- Gaussian observations, K = 2, fixed temperature
