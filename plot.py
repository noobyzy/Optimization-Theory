import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

from matplotlib import rcParams
rcParams.update({'font.size': 18, 'text.usetex': True})

linreg_GD_weights, linreg_GD_objective= pkl.load(open('./results/linreg_GD.pkl', 'rb'))
linreg_Newton_weights, linreg_Newton_objective= pkl.load(open('./results/linreg_Newton.pkl', 'rb'))
logreg_GD_weights, logreg_GD_objective= pkl.load(open('./results/logreg_GD.pkl', 'rb'))
logreg_Newton_weights, logreg_Newton_objective= pkl.load(open('./results/logreg_Newton.pkl', 'rb'))

def plot_linreg():
    '''
    linear regression: weights
    '''
    linreg_dimension = 8
    plt.figure()
    plt.plot(range(len(linreg_GD_weights)), np.array(linreg_GD_weights) / np.sqrt(linreg_dimension), label = 'GD')
    plt.plot(range(len(linreg_Newton_weights)), np.array(linreg_Newton_weights) / np.sqrt(linreg_dimension), label = 'Newton')

    plt.hlines(np.array([1e-2, 1e-3, 1e-4]), xmin=0, xmax=len(linreg_GD_weights)-1, linestyles='dashed', colors='#89CE94')
    plt.vlines(np.array([5, 8, 11]), ymin=[5e-3, 5e-4, 5e-5], ymax=[5e-2, 5e-3, 5e-4], linestyles='dashdot', colors='#C4B7CB')
    plt.vlines(np.array([124, 222, 321]), ymin=[5e-3, 5e-4, 5e-5], ymax=[5e-2, 5e-3, 5e-4], linestyles='dashdot', colors='#B7245C')

    plt.text(x=5, y=1e-2, s='5')
    plt.text(x=8, y=1e-3, s='8')
    plt.text(x=11, y=1e-4, s='11')

    plt.text(x=124, y=1e-2, s='124')
    plt.text(x=222, y=1e-3, s='222')
    plt.text(x=321, y=1e-4, s='321')

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$\frac{1}{\sqrt{d}}\|x^{(k)}-x^{\star}\|_2$')
    plt.title('Linear Regression')
  
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./results/linreg_weights.png', dpi=1200)
    plt.show()


    '''
    linear regression: objective
    '''

    plt.figure()
    plt.plot(range(len(linreg_GD_objective)), linreg_GD_objective, label = 'GD')
    plt.plot(range(len(linreg_Newton_objective)), linreg_Newton_objective, label = 'Newton')

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$f(x^{(k)}) - p^{\star}$')
    plt.title('Linear Regression')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./results/linreg_objective.png', dpi=1200)
    plt.show()

def plot_logreg():
    '''
    logistic regression: weights
    '''
    logreg_dimension = 785
    plt.figure()
    plt.plot(range(len(logreg_GD_weights)), np.array(logreg_GD_weights) / np.sqrt(logreg_dimension), label = 'GD')
    plt.plot(range(len(logreg_Newton_weights)), np.array(logreg_Newton_weights) / np.sqrt(logreg_dimension), label = 'Newton')

    plt.hlines(np.array([1e-2, 1e-3, 1e-4]), xmin=0, xmax=len(logreg_GD_weights)-1, linestyles='dashed', colors='#89CE94')
    plt.vlines(np.array([10, 23, 33]), ymin=[5e-3, 5e-4, 5e-5], ymax=[5e-2, 5e-3, 5e-4], linestyles='dashdot', colors='#C4B7CB')
    plt.vlines(np.array([7, 62, 149]), ymin=[5e-3, 5e-4, 5e-5], ymax=[5e-2, 5e-3, 5e-4], linestyles='dashdot', colors='#B7245C')

    plt.text(x=10, y=1e-2, s='10')
    plt.text(x=23, y=1e-3, s='23')
    plt.text(x=33, y=1e-4, s='33')

    plt.text(x=7, y=1e-2, s='7')
    plt.text(x=62, y=1e-3, s='62')
    plt.text(x=149, y=1e-4, s='149')

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$\frac{1}{\sqrt{d}}\|x^{(k)}-x^{\star}\|_2$')
    plt.title('Logistic Regression')
  
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./results/logreg_weights.png', dpi=1200)
    plt.show()


    '''
    logistic regression: objective
    '''

    plt.figure()
    plt.plot(range(len(logreg_GD_objective)), logreg_GD_objective, label = 'GD')
    plt.plot(range(len(logreg_Newton_objective)), logreg_Newton_objective, label = 'Newton')

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$f(x^{(k)}) - p^{\star}$')
    plt.title('Logistic Regression')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./results/logreg_objective.png', dpi=1200)
    #plt.show()

if __name__ == '__main__':
    
    #plot_linreg()
    plot_logreg()
    
    

    