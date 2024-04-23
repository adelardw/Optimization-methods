import scipy
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

from optimization import *
from oracles import *
from utils import *


def Oracles(A, b, regcoef=0, orcl=QuadraticOracle):
    return orcl(A, b, regcoef) if orcl==create_log_reg_oracle else orcl(A, b)

class experiment1_init:

    def __init__(self, n, k):
        ran = np.random.uniform(1, k, n)
        self.A = scipy.sparse.diags(ran)/k
        self.b = np.random.rand(n)


def experiment1_res(pow_n, max_k, stepk ):

    nvect = []
    kvect = np.arange(20, max_k, stepk) 
    for i in range(1,pow_n):
        nvect.append(10**(i))
    
    iterations = 10
    T = {} 
    for i in nvect: 
        T[i] = [[] for _ in range(iterations)] 
        for j in range(iterations): 
            for k in kvect: 
                init = experiment1_init(i, k) 
                A = init.A
                b = init.b
                oracle = Oracles(A, b)
                matvec = lambda x: oracle.hess_vec(np.zeros(i), x)
                [x_star, msg, history] = conjugate_gradients(matvec, b, np.zeros(i), tolerance=1e-4, trace=True, display=False)
                T[i][j].append(len(history['residual_norm']))
            
            

            plt.plot(kvect, T[i][j])
        plt.plot(kvect, np.zeros_like(kvect),label=f'n = {i}')
    plt.ylabel('T(n,k)') 
    plt.xlabel(r'κ') 
    plt.grid()
    plt.title('assemblage of curves')
    plt.legend() 
    plt.show()
    

def call_exp1():
    pow_ten = 4
    max_k = 1000
    stepk = 100

    experiment1_res(pow_ten, max_k, stepk=stepk)

#call_exp1()

class experiment2_init:

    def __init__(self):
        self.gessete = load_svmlight_file('gisette_scale')
        self.news20 = load_svmlight_file('news20.scale')

def experiment2():
    initDatasets = experiment2_init()
    gessete = initDatasets.gessete
    news20 = initDatasets.news20

    memory_sizes = [0, 1, 5, 10, 50, 100]
    logreg_gessete = Oracles(gessete[0], gessete[1], regcoef=1/gessete[0].shape[0], orcl=create_log_reg_oracle)
    logreg_news20 = Oracles(news20[0], news20[1], regcoef=1/news20[0].shape[0], orcl=create_log_reg_oracle)
    x_0_gessete = np.zeros(gessete[0].shape[1])
    x_0_news20 = np.zeros(news20[0].shape[1])


    _, ax = plt.subplots(2, 2, figsize=(16, 16))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for memory in memory_sizes:
        _, _, gessete_res = lbfgs(logreg_gessete, x_0_gessete, tolerance=1e-4, max_iter=20, memory_size=memory,
                                  line_search_options=None, display=False, trace=True)
        _, _, news20_res = lbfgs(logreg_news20, x_0_news20, tolerance=1e-4, max_iter=20, memory_size=memory,
                                  line_search_options=None, display=False, trace=True)
        
        ax[0][0].semilogy(np.arange(len(gessete_res['grad_norm'])), np.power(gessete_res['grad_norm'],2)/gessete_res['grad_norm'][0]**2, label = f'(memory size ={memory})')
        ax[0][0].set_title('Gessete (Num Iterations)')
        ax[0][0].set(xlabel="Num Iterations", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
        ax[0][0].grid()
        ax[0][0].legend()

        ax[0][1].semilogy(gessete_res['time'], np.power(gessete_res['grad_norm'],2)/gessete_res['grad_norm'][0]**2, label = f'(memory size ={memory})')
        ax[0][1].set_title('Gessete (Real Time)')
        ax[0][1].set(xlabel="Real Time", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
        ax[0][1].grid()        
        ax[0][1].legend()
        
        ax[1][0].semilogy(np.arange(len(news20_res['grad_norm'])), np.power(news20_res['grad_norm'],2)/news20_res['grad_norm'][0]**2, label = f'(memory size ={memory})')
        ax[1][0].set_title('News20 (Num Iterations)')
        ax[1][0].set(xlabel="Num Iterations", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
        ax[1][0].grid()
        ax[1][0].legend()

        ax[1][1].semilogy(news20_res['time'], np.power(news20_res['grad_norm'],2)/news20_res['grad_norm'][0]**2, label = f'(memory size ={memory})')
        ax[1][1].set_title('News20 (Real Time)')
        ax[1][1].set(xlabel="Real Time", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
        ax[1][1].grid()
        ax[1][1].legend()
    
    plt.show()


#experiment2()

class experiment3_init:

    def __init__(self):
        init_gessete_news_20 = experiment2_init()
        self.gessete = init_gessete_news_20.gessete
        self.news20 = init_gessete_news_20.news20
        self.realsim = load_svmlight_file('real-sim')
        self.w8a = load_svmlight_file('w8a')
        self.rcv = load_svmlight_file('rcv1_train.binary')


def experiment3():
    
    initDatasets = experiment3_init()
    gessete = initDatasets.gessete
    news20 = initDatasets.news20
    realsim = initDatasets.realsim
    w8a = initDatasets.w8a
    rcv = initDatasets.rcv
    
    logreg_gessete = Oracles(gessete[0], gessete[1], regcoef=1/gessete[0].shape[0], orcl=create_log_reg_oracle)
    logreg_news20 = Oracles(news20[0], news20[1], regcoef=1/news20[0].shape[0], orcl=create_log_reg_oracle)
    logreg_realsim = Oracles(realsim[0], realsim[1], regcoef=1/realsim[0].shape[0], orcl=create_log_reg_oracle)
    logreg_w8a = Oracles(w8a[0], w8a[1], regcoef=1/w8a[0].shape[0], orcl=create_log_reg_oracle)
    logreg_rcv = Oracles(rcv[0], rcv[1], regcoef=1/rcv[0].shape[0], orcl=create_log_reg_oracle)

    x_0_gessete = np.zeros(gessete[0].shape[1])
    x_0_news20 = np.zeros(news20[0].shape[1])
    x_0_realsim = np.zeros(realsim[0].shape[1])
    x_0_w8a = np.zeros(w8a[0].shape[1])
    x_0_rcv = np.zeros(rcv[0].shape[1])
    
    print(x_0_gessete.shape, x_0_news20.shape, x_0_realsim.shape, x_0_w8a.shape, x_0_rcv.shape)

    _, ax = plt.subplots(2, 5, figsize=(16, 16))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    _,_, gessete_lbfgs = lbfgs(logreg_gessete, x_0_gessete, tolerance=1e-4, max_iter=1000, memory_size=10,
                               line_search_options=None, display=False, trace=True)
    

    _, _, gessete_newton = hessian_free_newton(logreg_gessete, x_0_gessete, tolerance=1e-5, max_iter=1000,
                                                line_search_options=None, display=False, trace=True)
    
    _,_, gessete_gd = gradient_descent(logreg_gessete, x_0_gessete, tolerance=1e-5, max_iter=1000,
                                       line_search_options=None, trace=True, display=False)

    ax[0][0].semilogy(np.arange(len(gessete_lbfgs['grad_norm'])), np.power(gessete_lbfgs['grad_norm'],2)/gessete_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[0][0].semilogy(np.arange(len(gessete_newton['grad_norm'])), np.power(gessete_newton['grad_norm'],2)/gessete_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[0][0].semilogy(np.arange(len(gessete_gd['grad_norm'])), np.power(gessete_gd['grad_norm'],2)/gessete_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[0][0].set_title('Gessete (Num Iterations)')
    ax[0][0].set(xlabel="Num Iterations", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[0][0].grid()
    ax[0][0].legend()

    ax[1][0].semilogy(gessete_lbfgs['time'], np.power(gessete_lbfgs['grad_norm'],2)/gessete_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[1][0].semilogy(gessete_newton['time'], np.power(gessete_newton['grad_norm'],2)/gessete_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[1][0].semilogy(gessete_gd['time'], np.power(gessete_gd['grad_norm'],2)/gessete_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[1][0].set_title('Gessete (Real Time)')
    ax[1][0].set(xlabel="Real Time", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[1][0].grid()
    ax[1][0].legend()
    

    
    _,_, news20_lbfgs = lbfgs(logreg_news20, x_0_news20, tolerance=1e-4, max_iter=1000, memory_size=10,
                              line_search_options=None, display=False, trace=True)
    
    _,_, news20_newton = hessian_free_newton(logreg_news20, x_0_news20, tolerance=1e-4, max_iter=1000, 
                           line_search_options=None, display=False, trace=True)
    
    _,_, news20_gd = gradient_descent(logreg_news20, x_0_news20, tolerance=1e-5, max_iter=1000,
                     line_search_options=None, trace=True, display=False)
    
    ax[0][1].semilogy(np.arange(len(news20_lbfgs['grad_norm'])), np.power(news20_lbfgs['grad_norm'],2)/news20_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[0][1].semilogy(np.arange(len(news20_newton['grad_norm'])), np.power(news20_newton['grad_norm'],2)/news20_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[0][1].semilogy(np.arange(len(news20_gd['grad_norm'])), np.power(news20_gd['grad_norm'],2)/news20_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[0][1].set_title('News20 (Num Iterations)')
    ax[0][1].set(xlabel="Num Iterations", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[0][1].grid()
    ax[0][1].legend()

    ax[1][1].semilogy(news20_lbfgs['time'], np.power(news20_lbfgs['grad_norm'],2)/news20_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[1][1].semilogy(news20_newton['time'], np.power(news20_newton['grad_norm'],2)/news20_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[1][1].semilogy(news20_gd['time'], np.power(news20_gd['grad_norm'],2)/news20_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[1][1].set_title('News20 (Real Time)')
    ax[1][1].set(xlabel="Real Time", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[1][1].grid()
    ax[1][1].legend()


    _,_, realsim_lbfgs = lbfgs(logreg_realsim, x_0_realsim, tolerance=1e-4, max_iter=1000, memory_size=10,
                              line_search_options=None, display=False, trace=True)
    
    _,_, realsim_newton = hessian_free_newton(logreg_realsim, x_0_realsim, tolerance=1e-4, max_iter=1000, 
                          line_search_options=None, display=False, trace=True)
    
    _,_, realsim_gd = gradient_descent(logreg_realsim, x_0_realsim, tolerance=1e-5, max_iter=1000,
                     line_search_options=None, trace=True, display=False)
    
    ax[0][2].semilogy(np.arange(len(realsim_lbfgs['grad_norm'])), np.power(realsim_lbfgs['grad_norm'],2)/realsim_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[0][2].semilogy(np.arange(len(realsim_newton['grad_norm'])), np.power(realsim_newton['grad_norm'],2)/realsim_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[0][2].semilogy(np.arange(len(realsim_gd['grad_norm'])), np.power(realsim_gd['grad_norm'],2)/realsim_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[0][2].set_title('Realsim (Num Iterations)')
    ax[0][2].set(xlabel="Num Iterations", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[0][2].grid()
    ax[0][2].legend()

    ax[1][2].semilogy(realsim_lbfgs['time'], np.power(realsim_lbfgs['grad_norm'],2)/realsim_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[1][2].semilogy(realsim_newton['time'], np.power(realsim_newton['grad_norm'],2)/realsim_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[1][2].semilogy(realsim_gd['time'], np.power(realsim_gd['grad_norm'],2)/realsim_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[1][2].set_title('Realsim (Real Time)')
    ax[1][2].set(xlabel="Real Time", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[1][2].grid()
    ax[1][2].legend()

    _,_, w8a_lbfgs = lbfgs(logreg_w8a, x_0_w8a, tolerance=1e-4, max_iter=1000, memory_size=10,
                              line_search_options=None, display=False, trace=True)
    
    _,_, w8a_newton= hessian_free_newton(logreg_w8a, x_0_w8a, tolerance=1e-4, max_iter=1000, 
                          line_search_options=None, display=False, trace=True)
    
    _,_, w8a_gd = gradient_descent(logreg_w8a, x_0_w8a, tolerance=1e-5, max_iter=1000,
                     line_search_options=None, trace=True, display=False)
    
    ax[0][3].semilogy(np.arange(len(w8a_lbfgs['grad_norm'])), np.power(w8a_lbfgs['grad_norm'],2)/w8a_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[0][3].semilogy(np.arange(len(w8a_newton['grad_norm'])), np.power(w8a_newton['grad_norm'],2)/w8a_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[0][3].semilogy(np.arange(len(w8a_gd['grad_norm'])), np.power(w8a_gd['grad_norm'],2)/w8a_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[0][3].set_title('w8a (Num Iterations)')
    ax[0][3].set(xlabel="Num Iterations", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[0][3].grid()
    ax[0][3].legend()

    ax[1][3].semilogy(w8a_lbfgs['time'], np.power(w8a_lbfgs['grad_norm'],2)/w8a_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[1][3].semilogy(w8a_newton['time'], np.power(w8a_newton['grad_norm'],2)/w8a_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[1][3].semilogy(w8a_gd['time'], np.power(w8a_gd['grad_norm'],2)/w8a_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[1][3].set_title('w8a (Real Time)')
    ax[1][3].set(xlabel="Real Time", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[1][3].grid()
    ax[1][3].legend()

    _,_, rcv_lbfgs = lbfgs(logreg_rcv, x_0_rcv, tolerance=1e-4, max_iter=1000, memory_size=10,
                            line_search_options=None, display=False, trace=True)
    
    _,_, rcv_newton= hessian_free_newton(logreg_rcv, x_0_rcv, tolerance=1e-4, max_iter=1000,
                                         line_search_options=None, display=False, trace=True)
    
    _,_, rcv_gd = gradient_descent(logreg_rcv, x_0_rcv, tolerance=1e-5, max_iter=1000,
                     line_search_options=None, trace=True, display=False)
    
    ax[0][4].semilogy(np.arange(len(rcv_lbfgs['grad_norm'])), np.power(rcv_lbfgs['grad_norm'],2)/rcv_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[0][4].semilogy(np.arange(len(rcv_newton['grad_norm'])), np.power(rcv_newton['grad_norm'],2)/rcv_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[0][4].semilogy(np.arange(len(rcv_gd['grad_norm'])), np.power(rcv_gd['grad_norm'],2)/rcv_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[0][4].set_title('RCV  (Num Iterations)')
    ax[0][4].set(xlabel="Num Iterations", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[0][4].grid()
    ax[0][4].legend()

    ax[1][4].semilogy(rcv_lbfgs['time'], np.power(rcv_lbfgs['grad_norm'],2)/rcv_lbfgs['grad_norm'][0]**2, label = f'(lbfgs)')
    ax[1][4].semilogy(rcv_newton['time'], np.power(rcv_newton['grad_norm'],2)/rcv_newton['grad_norm'][0]**2, label = f'(hess newton)')
    ax[1][4].semilogy(rcv_gd['time'], np.power(rcv_gd['grad_norm'],2)/rcv_gd['grad_norm'][0]**2, label = f'(grad descent)')
    ax[1][4].set_title('RCV  (Real Time)')
    ax[1][4].set(xlabel="Real Time", ylabel=r'||∇f(x)||**2/ ||∇f(x0)||**2')
    ax[1][4].grid()
    ax[1][4].legend()
    
    plt.show()

experiment3()


