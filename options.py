import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'cali',
                        help="dataset for different task")
    parser.add_argument('--model', type=str, default = 'linreg',
                        help="using linear regression or logistic regression")
    parser.add_argument('--lr', type=float, default = 0.5,
                        help="learning rate for each update step")
    parser.add_argument('--optimizer', type=str, default = 'GD',
                        help="using GD or Newton for update")
    parser.add_argument('--iteration', type=int, default = 500,
                        help="maximum update iterations if not exit automatically")
    '''
    ONLY: for logistic regression 
    '''
    parser.add_argument('--augment', type=bool, default = False,
                        help="augment the sample with 1")
    parser.add_argument('--target_ind_0', type=int, default = 0,
                        help="the first target label, will be set to 0")
    parser.add_argument('--target_ind_1', type=int, default = 1,
                        help="the second target label, will be set to 1")
    parser.add_argument('--gamma', type=float, default = 0.2,
                        help="penalty term for logistic regression")
    


    args = parser.parse_args()
    return args
