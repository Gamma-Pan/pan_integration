## Polynomial Approximation Numerical Integration (working title)

Material regarding a new numerical ODE solver I develop in the context of my thesis and its application to Neural ODEs.

Our goal is to create a parallel-in-time solver that leverages modern GPU architectures to address the bottlenecks of 
continuous time ML models. Renewed interest in these models was re-ignited by [[1](https://arxiv.org/abs/1806.07366)].

For an implementation of Neural DEs in PyTorch see [[2](https://github.com/DiffEqML/torchdyn)]

### In this repo:
- API for the PAN solver
  - With adjoint support
- A batched Newton implementation with line search
- A PyTorch implementation of the modified Cholesky Factorization from Gill, Murray, Wright - Practical Optimization
- Experiments comparing PAN with other commonly used solvers
- Thesis Latex code and pdf
- Matplotlib plots source

### wandb
experiment tracking here: https://wandb.ai/gamma_pan/pan_integration?nw=nwusergamma_pan

### Lorenz attractor, system solved using PAN Integration.

https://github.com/Gamma-Pan/pan_integration/assets/84142041/d079a481-9986-4f82-8a40-d148d3550115

