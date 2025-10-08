# Martingale Posterior Neural Networks for Fast Sequential Decision Making

![hilofi-train](https://github.com/user-attachments/assets/d7c4e6ce-4389-4a94-816f-246491c17148)


## Citation
```bib
@article{duran2025scalable,
  title={Martingale Posterior Neural Networks for Fast Sequential Decision Making},
  author={Duran-Martin, Gerardo and S{\'a}nchez-Betancourt, Leandro and Cartea, {\'A}lvaro and Murphy, Kevin},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2025}
}
```

## Overview
The code is divided into notebooks and scripts.
To reproduce the experiments, run the following notebooks

## Reproducing Bayesian optimisation results
Run [`bayesopt-classic`](./experiments/01-bayesopt-classic.ipynb) notebook.

## Reproducing recommender system results
Run [`recommender-systems`](./experiments/02-recommender-systems.ipynb) notebook.

## Reproducing MNIST for classification results
Run [`mnist-online-classification`](./experiments/04-mnist-online-classification.ipynb) notebook.

## Reproducing MNIST as a neural contextual bandit problem
Run [`mnist-bandit-analysis`](./experiments/03-mnist-bandit-analysis.ipynb) notebook.

## Run in-between uncertainty results
Run [`toy-in-between`](./experiments/05-toy-in-between.ipynb) notebook.

# Dependencies

```
bayesian-optimization==2.0.3
click==8.1.8
distrax==0.1.5
flax==0.10.3
jax==0.5.1
jaxlib==0.5.1
toml==0.10.2
numpy==2.2.3
vbll==0.4.6
matplotlib==3.10.0
```
