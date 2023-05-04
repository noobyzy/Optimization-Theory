import numpy as np
import cvxpy as cp


class LinearRegression():
    def __init__(self, args, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
        self.num_samples = self.X_train.shape[0]
        self.dimension = self.X_train.shape[1]

        self.weights = np.zeros_like(X_train[0])

        self.lr = args.lr
        self.optimizer = args.optimizer

        print("============= CVX solving =============")
        self.opt_weights, self.opt_obj = self.CVXsolve()
        print("============= CVX solved =============")

    
        self.pre_Hessian = self.Hessian(self.weights)
    
    def objective(self, weights):
        '''
        return the objective value of the problem
        note that the objective is averaged over all samples
        '''
        return 0.5 * np.linalg.norm(self.X_train @ weights - self.Y_train, ord=2) ** 2 / self.num_samples

    def gradient(self, weights):
        '''
        return the gradient of objective function
        note that the gradient is averaged over all samples
        '''
        return self.X_train.T @ (self.X_train @ weights - self.Y_train) / self.num_samples

    def Hessian(self, weights):
        '''
        return the Hessian of objective function
        note that the Hessian is averaged over all samples
        '''
        return self.X_train.T @ self.X_train / self.num_samples


    def update(self):
        '''
        update model weights using GD / Newton step
        '''
        gradient = self.gradient(self.weights)
        
        if self.optimizer == 'GD':
            update_direction = gradient
        elif self.optimizer == 'Newton':
            update_direction = np.linalg.solve(self.pre_Hessian, gradient)
        else:
            raise NotImplementedError

        self.weights -= self.lr * update_direction

        return self.diff_cal(self.weights)

    def CVXsolve(self):
        '''
        use CVXPY to solve optimal solution
        '''
        x = cp.Variable(self.dimension)
        objective = cp.sum_squares(self.X_train @ x - self.Y_train)
        prob = cp.Problem(cp.Minimize(objective))
        prob.solve(solver=cp.MOSEK)

        opt_weights = np.array(x.value)
        opt_obj = self.objective(opt_weights)

        return opt_weights, opt_obj

    def diff_cal(self, weights):
        '''
        calculate the difference of input model weights with optimal in terms of:
        -   weights
        -   objective
        '''
        weight_diff = np.linalg.norm(weights - self.opt_weights)
        obj_diff = abs(self.objective(weights) - self.opt_obj) 
        return weight_diff, obj_diff