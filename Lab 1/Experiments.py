from oracles import *
from optimization import *
from plot_trajectory_2d import *
from sklearn.datasets import load_svmlight_file
from typing import Callable



def Oracles(A, b, regcoef=0, orcl=QuadraticOracle):
    return orcl(A, b, regcoef) if orcl==create_log_reg_oracle else orcl(A, b)


class experiment1_init:
    """
    Эксперимент: Траектория градиентного спуска на квадратичной функции
    Проанализируйте траекторию градиентного спуска для нескольких квадра-
    тичных функций: придумайте две-три квадратичные двумерные функции,
    на которых работа метода будет отличаться, нарисуйте графики с линиями
    уровня функций и траекториями методов.
    Попробуйте ответить на следующий вопрос: Как отличается поведе-
    ние метода в зависимости от числа обусловленности функции, выбора на-
    чальной точки и стратегии выбора шага (константная стратегия, Армихо,
    Вульф)?
    Для рисования линий уровня можете воспользоваться функцией plot_levels,
    a  для рисования траекторий - plot_trajectory из файла plot_trajectory_2d.py,
    прилагающегося к заданию.
    """

    def __init__(self):
        np.random.seed(32)
        A = np.random.uniform(0, np.sqrt(13), (2, 3))
        A = np.sqrt((A.T @ A + (A.T @ A).T))/2
        A1, A2, A3 = A[:2, :2], A[1:3, 1:3:1], np.diag(np.diag(A))[:2, :2]
        b = np.random.randint(-3, 2, size=(2, 3))

        b1, b2, b3 = b[:, 0], b[:, 1], b[:, 2]


        self.f1 = Oracles(A1, b1)
        self.f2 = Oracles(A2, b2)
        self.f3 = Oracles(A3, b3)


    def realize_and_show(self, start_pts):

        def experiment1(func, method, nfig, n_func, options=None):
            collect = []
            for u in start_pts:
                if method == 'gradient descent':
                    [x_star, msg, history] = gradient_descent(func, u, trace=True, line_search_options=options)
                    collect.append(x_star)
                if method=='newton':
                    [x_star, msg, history] = newton(func, u, trace=True, line_search_options=options)
                    collect.append(x_star)

                plt.figure(nfig)
                plot_levels(func.func)
                plot_trajectory(func.func, history['x'], label=u)
                plt.grid()
                color = 'red' if u==start_pts[0] else 'green'
                plt.gca().get_lines()[0].set_color(color)
                plt.legend()
                plt.title(f'{method} \n  {options} \n start: {u} \n n_func f{n_func + 1}')
            
                

        line_search_options_0 = {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'alpha_0': 1}
        line_search_options_1 = {'method': 'Armijo', 'c1': 1e-4}
        line_search_options_2 = {'method': 'Constant', 'c': 1e-4}

        
        options = [line_search_options_0, line_search_options_1, line_search_options_2]
        methods = ['gradient descent']
        functions = [self.f1, self.f2, self.f3]

        for i,opt in enumerate(options):
            for j, m in enumerate(methods):
                for k, func in enumerate(functions):
                    num = f'{(i + j + k)}'
                    k += 1
                    experiment1(func, m, nfig = num, n_func=k-1, options=opt)
                plt.ylabel('y')
                plt.xlabel('x')
                plt.grid()
                plt.show()
 


def call_exp1():
    pts = [[1, 2], [np.pi, -np.pi]]
    A = experiment1_init()
    return A.realize_and_show(pts)

#call_exp1()


class experiment2_init:
    '''
    Эксперимент: Зависимость числа итераций градиентного спуска от числа
    обусловленности и размерности пространства

    Исследуйте, как зависит число итераций, необходимое градиентному спуску
    для сходимости, от следующих двух параметров: 1) числа обусловленности
    κ ≥ 1 оптимизируемой функции и 2) размерности пространства n оптими-
    зируемых переменных.
    Для этого для заданных параметров n и κ сгенерируйте случайным об-
    разом квадратичную задачу размера n с числом обусловленности κ и за-
    пустите на ней градиентный спуск с некоторой фиксированной требуемой
    точностью. Замерьте число итераций T(n, κ), которое потребовалось сде-
    лать методу до сходимости (успешному выходу по критерию остановки).

    Рекомендация: Проще всего сгенерировать случайную квадратичную
    задачу размера n с заданным числом обусловленности κ следующим об-
    разом. В качестве матрицы A ∈ Sn
    ++ удобно взять просто диагональную
    матрицу A = Diag(a), у которой диагональные элементы сгенерированы
    случайно в пределах [1, κ], причем min(a) = 1, max(a) = κ. В качестве век-
    тора b ∈ Rn можно взять вектор со случайными элементами. Диагональные
    матрицы удобно рассматривать, поскольку с ними можно эффективно ра-
    ботать даже при больших значениях n. Рекомендуется хранить матрицу A
    в формате разреженной диагональной матрицы (см. scipy.sparse.diags).
    Зафиксируйте некоторое значение размерности n. Переберите различ-
    ные числа обусловленности κ по сетке и постройте график зависимости
    T(κ, n) от κ. Поскольку каждый раз квадратичная задача генерируется
    случайным образом, то повторите этот эксперимент несколько раз. В ре-
    зультате для фиксированного значения n у Вас должно получиться целое
    семейство кривых зависимости T(κ, n) от κ. Нарисуйте все эти кривые од-
    ним и тем же цветом для наглядности (например, красным).
    Теперь увеличьте значение n и повторите эксперимент снова. Вы долж-
    ны получить новое семейство кривых T (n′, κ) против κ. Нарисуйте их все
    одним и тем же цветом, но отличным от предыдущего (например, синим).

    Повторите эту процедуру несколько раз для других значений n. В итоге
    должно получиться несколько разных семейств кривых - часть красных (со-
    ответствующих одному значению n ), часть синих (соответствующих дру-
    гому значению n ), часть зеленых и т. д.
    Обратите внимание, что значения размерности n имеет смысл переби-
    рать по логарифмической сетке (например, n = 10, n = 100, n = 1000 и т.
    д.).
    '''

    def __init__(self, n, k):
        ran = np.random.uniform(1, k, n)
        self.A = scipy.sparse.diags(ran)/k
        self.b = np.random.rand(n)


def experiment2_res(pow_n, max_k, stepk ):

    nvect = []
    kvect = np.arange(20, max_k, stepk) 
    for i in range(1,pow_n):
        nvect.append(10**(i))
    
    iterations = 13 
    T = {} 
    for i in nvect: 
        T[i] = [[] for _ in range(iterations)] 
        for j in range(iterations): 
            for k in kvect: 
                init = experiment2_init(i, k) 
                A = init.A
                b = init.b
                oracle = Oracles(A, b) 
                [x_star, msg, history] = gradient_descent(oracle, np.zeros(i), trace=True) 
                T[i][j].append(len(history['grad_norm']))
            
            

            plt.plot(kvect, T[i][j])
        plt.plot(kvect, np.zeros_like(kvect),label=f'n = {i}')
    plt.ylabel('T(n,k)') 
    plt.xlabel(r'κ') 
    plt.grid()
    plt.title('assemblage of curves')
    plt.legend() 
    plt.show()
    

def call_exp2():
    pow_ten = 5
    max_k = 700
    stepk = 100

    experiment2_res(pow_ten, max_k, stepk=stepk)

class experiment3_init:
    '''
    Сравнить методы градиентного спуска и Ньютона на задаче обучения ло-
    гистической регрессии на реальных данных. B качестве реальных данных
    используйте следующие три набора с сайта LIBSVM3 w8a, gisette и real-sim.
    Коэффициент регуляризации взять стандартным образом: λ = 1/m.
    Параметры обоих методов взять равными параметрам по умолчанию.
    Начальную точку выбрать x0 = 0.
    Построить графики сходимости следующих двух видов:
    (а) Зависимость значения функции от реального времени работы мето-
    да.
    (b) Зависимость относительного квадрата нормы градиента ∥∇f (xk)∥2
    2 / ∥∇f (x0)∥2
    2
    (в логарифмической шкале) против реального времени работы.
    При этом оба метода (градиентный спуск и Ньютон) нужно рисовать на
    одном и том же графике.
    Укажите в отчете, какова стоимость итерации и сколько памяти требу-
    ется каждому из методов в зависимости от параметров m (размер выбор-
    ки) и n (размерность пространства). При оценке используйте нотацию O(·),
    скрывающую внутри себя абсолютные константы.
    Какие выводы можно сделать по результатам этого эксперимента? Ка-
    кой из методов лучше и в каких ситуациях?
    Рекомендация: Любой набор данных с сайта LIBSVM представляет
    из себя текстовый файл в формате svmlight. Чтобы считать такой тек-
    стовый файл, можно использовать функцию load_svmlight_file из модуля
    sklearn.datasets. Обратите внимание, что эта функция возвращает матрицу
    в формате scipy.sparse.csr_matrix, поэтому Ваша реализация логистического
    оракула должна поддерживать такие матрицы.
    '''

    def __init__(self):
        w8a = load_svmlight_file('w8a')
        gissete = load_svmlight_file('gisette_scale')
        real_sim = load_svmlight_file('real-sim')

        self.A1, self.A2, self.A3 = w8a[0], gissete[0], real_sim[0]
        self.b1, self.b2, self.b3 = w8a[1], gissete[1], real_sim[1]


def init_exp4(num_of_data = 2):
    init_data = experiment3_init()
    datsets = init_data.A1, init_data.A2, init_data.A3 
    coeficients = init_data.b1, init_data.b2, init_data.b3

    A = datsets[num_of_data]
    b = coeficients[num_of_data]

    oracle = Oracles(A, b, regcoef=1/A.shape[0], orcl=create_log_reg_oracle)
    _, _, historygd = gradient_descent(oracle, np.zeros(A.shape[1]), max_iter=4000, trace=True)
    _, _, historynew = newton(oracle, np.zeros(A.shape[1]),max_iter=4000, trace=True)

    return historygd, historynew

def show_func(grad, newt): 
    plt.plot(grad['time'], grad['func'])
    plt.plot(newt['time'], newt['func'])
    plt.xlabel("t [sec] ", fontsize=13) 
    plt.ylabel("f(x, t)", fontsize=13) 
    plt.legend(["Gradient descent", "Newton"])
    plt.title('Зависимость значения функции от реального времени работы метода.') 
    plt.grid()
    plt.show() 
 
def show_grad(grad, newt): 

    plt.semilogy(grad['time'], np.array(grad['grad_norm'])**2 / grad['grad_norm'][0]**2)
    plt.semilogy(newt['time'], np.array(newt['grad_norm'])**2 / newt['grad_norm'][0]**2)
    plt.xlabel("t [sec]", fontsize=13) 
    plt.ylabel(r'||∇f(x)||**2/ ||∇f(x0)||**2', fontsize=13) 
    plt.legend(["Gradient descent", "Newton"])
    plt.grid()
    plt.title('Зависимость относительного квадрата нормы градиента.') 
    


def call_exp3(num_of_data):
    historygd, historynew = init_exp4(num_of_data = num_of_data)
    plt.figure(0)
    show_func(historygd, historynew)
    plt.show()
    plt.figure(1)
    show_grad(historygd, historynew)
    plt.show()

    
    
def experiment45(c:np.array, back_c:np.array,
                 c1:np.array, c2:np.array,
                 optim: Callable = gradient_descent,
                 maxiter:int=25, n: int=2, method:str='gd'):


    '''
    Исследовать, как зависит поведение метода от стратегии подбора шага: кон-
    стантный шаг (попробовать различные значения), бэктрэкинг (попробовать
    различные константы c ), условия Вульфа (попробовать различные пара-
    метры c2).
    Рассмотрите квадратичную функцию и логистическую регрессию с мо-
    дельными данным (сгенерированными случайно).

    Запустите для этих функций градиентный спуск с разными стратегиями
    выбора шага из одной и той же начальной точки.
    Нарисуйте кривые сходимости (относительная невязка по функции в
    логарифмической шкале против числа итераций - для квадратичной функ-
    ции, относительный квадрат нормы градиента в логарифмической шкале
    против числа итераций - для логистической регрессии) для разных страте-
    гий на одном графике.
    Попробуйте разные начальные точки. Ответьте на вопрос: Какая стра-
    тегия выбора шага является самой лучшей?
    '''

    np.random.seed(13)
    ran = np.random.uniform(1, 100, n)
    A = np.diag(ran) #scipy.sparse.diags(ran)
    b = np.random.uniform(1, 100, n)
    b_log = np.random.randint(0, 2, n)
    b_log[b_log ==0 ] = -1

    oracle_quad = Oracles(A, b, regcoef=1/A.shape[0], orcl=create_log_reg_oracle)
    oracle_log2rec = Oracles(A, b, regcoef=1/A.shape[0], orcl=create_log_reg_oracle)



    res1 = []
    pts = [np.random.rand(n)] #,
           #np.zeros(n)]
           #np.ones(n),
           #p.random.uniform(0, 0.1, n), np.random.uniform(-100, 100, n)]

    if optim == gradient_descent:
       name_optim = 'GD'
    else:
       name_optim = 'Newton'


    for c_ in c:
      for i, p in enumerate(pts):

        line_search_options = {'method': 'Constant', 'c': c_}
        [x_opt, msg, history] = optim(oracle_quad, p,
                                        max_iter=maxiter,
                                        trace=True,
                                        line_search_options=line_search_options)


        iter1 = np.arange(len(history['func']))
        res1 = (
           np.log(
           np.abs(
           (np.array(history['func']) - np.array(history['func'][0])) / history['func'][0])))

        plt.figure(1)
        plt.plot(iter1, res1, label = f'point number {i}, \n parameter {c_}')
        plt.xlabel('N iterations')
        plt.ylabel('log(abs([F(x) -F(x0)]/F[x0]))')

        plt.legend()
        plt.title(f"{name_optim} \n {line_search_options['method']}")
    plt.grid()
    plt.show()


    for i, p in enumerate(pts):
      for c_ in back_c:


        line_search_options = {'method': 'Armijo', 'c1': c_}
        [x_opt, msg, history] = optim(oracle_quad, p,
                                      max_iter=maxiter,
                                      trace=True,
                                      line_search_options=line_search_options)


        iter1 = np.arange(len(history['func']))
        res1 = (
           np.log(
           np.abs(
           (np.array(history['func']) - np.array(history['func'][0])) / history['func'][0])))

        plt.figure(2)
        plt.plot(iter1, res1, label = f'point number {i}, \n parameter {c_}')
        plt.xlabel('N iterations')
        plt.ylabel('log(abs([F(x) -F(x0)]/F[x0]))')
        plt.legend()
        plt.title(f"{name_optim} \n {line_search_options['method']}")
    plt.grid()
    plt.show()


    for c1_, c2_ in zip(c1,c2):

      for i, p in enumerate(pts):


        line_search_options = {'method': 'Wolfe', 'c1': c1_, 'c2': c2_, 'alpha_0': 1}

        [x_opt, msg, history] = optim(oracle_quad, p,
                                      max_iter=maxiter, trace=True,
                                      line_search_options=line_search_options)
        
        iter1 = np.arange(len(history['func']))
        res1 = (
           np.log(
           np.abs(
           (np.array(history['func']) - np.array(history['func'][0])) / history['func'][0])))

        plt.figure(4)
        plt.plot(iter1, res1, label = f'point number {i}, \n parameters {c1_},{c2_}')
        plt.xlabel('N iterations')
        plt.ylabel('log(abs([F(x) -F(x0)]/F[x0]))')
        plt.legend()
        plt.title(f"{name_optim} \n {line_search_options['method']}")
    plt.grid()
    plt.show()


    for c_ in c:
      for i, p in enumerate(pts):
      
        line_search_options = {'method': 'Constant', 'c': c_}
        [x_opt, msg, history] = optim(oracle_log2rec, p,
                                      max_iter=maxiter,
                                      trace=True,
                                      line_search_options=line_search_options)

        iter1 = np.arange(len(history['func']))
        res1 = (
           np.log(
           np.array(history['grad_norm'])**2 /history['grad_norm'][0]**2))


        plt.figure(4)
        plt.plot(iter1, res1, label = f'point number {i}, \n parameters {c_}')
        plt.xlabel('N iterations')
        plt.ylabel('log(||grad(x||**2/||grad(x_start)||**2')
        plt.legend()
        plt.title(f"{name_optim} \n {line_search_options['method']}")
    plt.grid()
    plt.show()


    for c_ in back_c:
      for i, p in enumerate(pts):

        line_search_options = {'method': 'Armijo', 'c1': c_}
        [x_opt, msg, history] = optim(oracle_log2rec, p,
                                      max_iter=maxiter, trace=True,
                                      line_search_options=line_search_options)


        iter1 = np.arange(len(history['func']))
        res1 = (
           np.log(
           np.array(history['grad_norm'])**2 /history['grad_norm'][0]**2))


        plt.figure(5)
        plt.plot(iter1, res1, label = f'point number {i}, \n parameters {c_}')
        plt.xlabel('N iterations')
        plt.ylabel('log(||grad(x||**2/||grad(x_start)||**2')
        plt.legend()
        plt.title(f"{name_optim} \n {line_search_options['method']}")
    plt.grid()
    plt.show()

 
    for c1_,c2_ in zip(c1,c2):
      for i, p in enumerate(pts):

        line_search_options = {'method': 'Wolfe', 'c1': c1_, 'c2': c2_, 'alpha_0': 0.01}

        [x_opt, msg, history] = optim(oracle_log2rec, p,
                                      max_iter=maxiter, trace=True,
                                      line_search_options=line_search_options)

        iter1 = np.arange(len(history['func']))
        res1 = (
           np.log(
           np.array(history['grad_norm'])**2 /history['grad_norm'][0]**2))


        plt.figure(6)
        plt.plot(iter1, res1, label = f'point number {i}, \n parameters {c1_},{c2_}')
        plt.xlabel('N iterations')
        plt.ylabel('log(||grad(x||**2/||grad(x_start)||**2')
        plt.legend()
        plt.title(f"{name_optim} \n {line_search_options['method']}")
    plt.grid()
    plt.show()

def call_exp45(optim = gradient_descent):
    c1 = [1e-2, 1e-3, 1e-4]
    c2 = [1e-2, 1e-3, 1e-4]
    c = [1e0, 1e-1, 1e-3, 1e-4]
    back_c = [1e-2, 1e-3, 1e-4]

    experiment45(c, back_c, c1, c2,n=20,maxiter=100, optim=optim)


def all_experiments(num_experiments, num_dataset=0):
    if num_experiments==1:
      call_exp1()

    if num_experiments==2:
        call_exp2()

    if num_experiments==3:
        if num_dataset==0:
            call_exp3(0)

        if num_dataset==1:
            call_exp3(1)
      
        if num_dataset==2:
            call_exp3(2)

    if num_experiments==4:
        call_exp45(optim=gradient_descent)
    
    if num_experiments==5:
       call_exp45(optim=newton)

all_experiments(num_experiments=3, num_dataset=1)
    
    