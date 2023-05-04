# Optimization-Theory
-----------------------
A python (numpy) implementation of linear / logistic regression models, optimized with Gradient Descent / Newton's method.
An comparison with optimal solution (solved by CVXPY) is also provided.

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