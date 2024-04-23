import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool
import time 
from time import time
from tabulate import tabulate

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0).reshape(-1)

    start = time()
    max_iter = max_iter if max_iter is not None else x_k.size
    b = b.reshape(-1)
    g = lambda x, b: matvec(x) - b
    g_k = g(x_0, b)
    d_k = -g_k

    times = time() - start

    if trace:
      history['time'] = [times]
      history['residual_norm'] = [np.linalg.norm(g_k)]
      if x_k.size <= 2:
            history['x'] = [x_k]


    for i in range(max_iter):

        if np.linalg.norm(g_k) <= tolerance *np.linalg.norm(b):
            return x_k, 'success', history

        
        x_k = x_k + d_k * (g_k @ g_k)/(matvec(d_k) @ d_k)

        g_knext = g(x_k, b)
        g_knext = g_knext.reshape(-1)
        d_k = -g_knext + d_k * (g_knext @ g_knext)/(g_k @ g_k)

        g_k = g_knext

        times = time() - start
        if trace:
          history['time'].append(times)
          history['residual_norm'].append(np.linalg.norm(g_k))
          if x_k.size <= 2:
            history['x'].append(x_k)

    if display:
        print(tabulate([history], headers="keys", tablefmt="github"))

    return x_k, 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)

    N = len(x_0)
    x = x_0.copy()
    s_list = []
    y_list = []
    rho_list = []


    time_start = time()

    func = oracle.func
    grad = oracle.grad

    times = time() - time_start

    if trace:
      history['time'] = [times]
      history['func'] = [func(x_0)]
      history['grad_norm'] = [np.linalg.norm(grad(x_0))]
      if x.size <= 2:
        history['x'] = [x]
    
    for _ in range(max_iter):
        fx = func(x)
        gradx = grad(x)

        if np.linalg.norm(gradx) < tolerance:
            return x,'success', history
        
        if len(s_list) == 0:
            p = -gradx
        else:
            q = gradx.copy()
            alpha = [0] * len(s_list)
            for i in range(len(s_list) - 1, -1, -1):
                alpha[i] = rho_list[i] * np.dot(s_list[i], q)
                q -= alpha[i] * y_list[i]
            r = q
            
            for i in range(len(s_list)):
                beta = rho_list[i] * np.dot(y_list[i], r)
                r += (alpha[i] - beta) * s_list[i]
            p = -r
        
        alpha = line_search_tool.line_search(oracle=oracle, x_k=x, d_k=p, previous_alpha=None)
        s = alpha * p
        x_prev = x
        x = x + s


        if len(s_list) == memory_size and memory_size !=0:
            s_list.pop(0)
            y_list.pop(0)
            rho_list.pop(0)
        
        s_list.append(s)
        y = grad(x) - gradx
        y_list.append(y)
        rho_list.append(1 / np.dot(y, s))

        times = time() - time_start
        if trace:
          
          history['time'].append(times)
          history['func'].append(func(x))
          history['grad_norm'].append(np.linalg.norm(grad(x)))
          if x.size <= 2:
            history['x'].append(x)

    return x,'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).reshape(-1)


    time_start = time()
    grad = oracle.grad
    func = oracle.func
    hess = oracle.hess
    grad_start = grad(x_0)

    times = time() - time_start

    if trace:
      history['time'] = [times]
      history['func'] = [func(x_0)]
      history['grad_norm'] = [np.linalg.norm(grad_start)]
      if x_k.size <= 2:
        history['x'] = [x_k]

    alpha = 1
    

    for i in range(max_iter):

        
        grad_k = grad(x_k)

        if np.linalg.norm(grad_k)**2 <= tolerance * np.linalg.norm(grad_start)**2:
           return x_k, 'success', history
        
        grad_norm_k = np.linalg.norm(grad_k)
        theta = min(0.5, np.sqrt(grad_norm_k))

        matvec = lambda x: oracle.hess_vec(x_k, x)
        d_k = conjugate_gradients(matvec, -grad_k, -grad_k, tolerance=theta)[0]

        while True:
            if grad_k @ d_k <0:
               break
            else:
                theta /= 10
                d_k = conjugate_gradients(matvec, -grad_k, d_k, tolerance=theta)[0]
                


        x_k = x_k + alpha * d_k
        times = time() - time_start
        if trace:
          history['time'].append(times)
          history['func'].append(func(x_k))
          history['grad_norm'].append(np.linalg.norm(grad(x_k)))
          if x_k.size <= 2:
            history['x'].append(x_k)

        alpha = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=d_k, previous_alpha=alpha)

    if display:
        print(tabulate([history], headers="keys", tablefmt="github"))

    return x_k, 'iterations_exceeded', history



def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """


    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    start = time()

    grad = oracle.grad
    func = oracle.func
    times = time() - start
    grad_start = grad(x_0)

    if trace:
      history['time'] = [times]
      history['func'] = [func(x_0)]
      history['grad_norm'] = [np.linalg.norm(grad_start)]
      if x_k.size <= 2:
            history['x'] = [x_k]

    try:
        alpha = line_search_tool.alpha_0
    except:
        alpha = 1

    for _ in range(max_iter):
        grad_f = grad(x_k)
        if np.linalg.norm(grad_f) <= np.sqrt(tolerance) * np.linalg.norm(grad_start):
            return x_k, 'success', history

        if  ~np.all(np.isfinite(x_k)) or ~np.all(np.isfinite(grad_f)):
            return x_k, 'computational_error', history


        
        alpha = line_search_tool.line_search(oracle=oracle, x_k=x_k, d_k=-grad_f, previous_alpha=alpha)
        x_k = x_k - alpha * grad_f
        times = time() - start
        if trace:
          history['time'].append(times)
          history['func'].append(func(x_k))
          history['grad_norm'].append(np.linalg.norm(grad_f))
          if x_k.size <= 2:
            history['x'].append(x_k)

    if display:
        print(tabulate([history], headers="keys", tablefmt="github"))
    return x_k, 'iterations_exceeded', history