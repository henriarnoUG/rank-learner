# Rank-Learner: Orthogonal Ranking of Treatment Effects

This repository contains the code accompanying the paper: ``Rank-Learner: Orthogonal Ranking of Treatment Effects.``

## Structure
- `data/`: notebooks for generating synthetic and semi-synthetic datasets.
- `library/`: core implementation (models, training loops, metrics, utilities).
- `nuisances.ipynb`: nuisance estimation.
- `cate-learning.ipynb`: standard CATE estimation baselines.
- `rankers.ipynb`: ranking methods, including the proposed **Rank-Learner**.

## Usage
The code requires Python 3 and standard scientific Python packages, including PyTorch, NumPy, SciPy, and scikit-learn. Run the notebooks in `data/` to generate datasets, then use the demo notebooks in the root directory to train and evaluate the models.
