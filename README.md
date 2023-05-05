# Optimization-Theory
-----------------------
A python (numpy) implementation of linear / logistic regression models, optimized with Gradient Descent / Newton's method.
An comparison with optimal solution (solved by CVXPY) is also provided.

## Installation
Install the following packages via `pip` or `conda`:
* numpy
* sklearn
* cvxpy
* pickle

Note that I use `MOSEK` solver for cvxpy (see [download link](https://www.mosek.com/downloads/) for more detail, and you also need a [license](https://www.mosek.com/products/academic-licenses/)), you can also use the default solver (e.g., CVXOPT).

In `plot.py`, I use latex features for rendering axis labels. To enable this, you need to download latex (see [Instruction](https://patrickyoussef.com/blog/latex-plots/)).

## Task Description
In this task I consider the two convex problems, **linear regression** and **logistic regression**.
### linear regression
A classical regression model. Check `models/linear_regression.py` for more details.

### logistic regression
A classical classification model. Check `models/logistic_regression.py` for more details.
Note that the objective is augmented with an L2 norm (otherwise cvx cannot solve it / cannot solve it efficiently).

## Dataset
* In this code, I use [MNIST](http://yann.lecun.com/exdb/mnist/) (784 features) for logistic regression (and pick 2 labels for binary classification), and [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)  (8 features)for linear regression.
* Datasets are stored in the folder `datasets/`.
* The detailed preprocessing of datasets is available in `datasets/data_preprocess.py`.

## RUN the code
In `scripts/`, I provide the following basic commands:
```shell
# linear regression + gradient descent
bash linreg_GD.sh

# linear regression + Newton's method
bash linreg_Newton.sh

# logistic regression + gradient descent
bash logreg_GD.sh

# logistic regression + Newton's method
bash logreg_Newton.sh
```
Check the results in `results/`.

## Results

<p align="center">
  <img alt="Light" src="https://github.com/noobyzy/Optimization-Theory/blob/main/results/linreg_weights.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/noobyzy/Optimization-Theory/blob/main/results/linreg_objective.pn)" width="45%">
</p>

<p align="center">
  <img alt="Light" src="https://github.com/noobyzy/Optimization-Theory/blob/main/results/logreg_weights.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/noobyzy/Optimization-Theory/blob/main/results/logreg_objective.pn)" width="45%">
</p>

