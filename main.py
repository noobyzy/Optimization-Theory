import sys
sys.path.append('./models')
sys.path.append('./datasets')

import os
import numpy as np
import pickle as pkl
from datasets.data_preprocess import data_preprocess
from models import linear_regression, logistic_regression
from options import args_parser

current_work_dir = os.path.dirname(__file__)

if __name__ == '__main__':
    '''
    arguments
    '''
    args = args_parser()

    '''
    dataset preprocess
    '''    
    (x_train, y_train), (x_test, y_test) = data_preprocess(args)

    '''
    detailed argument setting for algorithm
    '''
    print('==========================')
    if args.model == 'linreg':
        print('Linear regression')
    elif args.model == 'logreg':
        print('Logistic regression')
    else:
        raise NotImplementedError
    print('learning rate: ', args.lr)
    print('Optimizer: ', args.optimizer)


    if args.model == 'linreg':
        Model = linear_regression.LinearRegression(args=args, X_train=x_train, Y_train=y_train)
    elif args.model == 'logreg':
        Model = logistic_regression.LogisticRegression(args=args, X_train=x_train, Y_train=y_train)
    else:
        raise NotImplementedError
    
    weight_diff_list = []
    obj_diff_list = []
    weight_diff, obj_diff = Model.diff_cal(Model.weights)
    print("\n------------ Initial ------------")
    print("weight error: {:.4e}".format(weight_diff))
    print("objective error: {:.4e}".format(obj_diff))

    Eigvals = np.linalg.eigvals(Model.pre_Hessian)
    print("\nmax eigenvalue of Hessian:{:.4f}".format(np.max(Eigvals)))
    print("min eigenvalue of Hessian:{:.4f}".format(np.min(Eigvals)))

    '''
    update
    '''
    for i in range(args.iteration):
        weight_diff, obj_diff = Model.update()
        print("\n------------ Iteration {} ------------".format(i+1))
        print("weight error: {:.4e}".format(weight_diff))
        print("objective error: {:.4e}".format(obj_diff))
        weight_diff_list.append(weight_diff)
        obj_diff_list.append(obj_diff)

        if weight_diff / np.sqrt(Model.dimension) <= 1e-5:
            break
    

    file_name = './results/{}_{}.pkl'.format(args.model, args.optimizer)
    file_name = os.path.join(current_work_dir, file_name)

    with open(file_name, 'wb') as f:
        pkl.dump([weight_diff_list, obj_diff_list], f)



    