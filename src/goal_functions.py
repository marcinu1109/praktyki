import numpy as np

def ackley(xs):
    a, b, c = 20, 0.2, 2*np.pi
    n = xs.size
    s1 = (xs**2).sum()
    s2 = np.cos(c*xs).sum()
    return -a*np.exp(-b*np.sqrt(1/n*s1))-np.exp(1/n*s2)+a+np.exp(1)

def griewank(xs):
    n = xs.size
    return (xs**2).sum()/4000 - (np.cos(xs/(np.sqrt(np.arange(n)+1)))).prod() + 1

def rastrigin(xs):
    n = xs.size
    return 10*n + (xs**2 - 10*np.cos(2*np.pi*xs)).sum()

def shwefel(xs):
    n = xs.size
    return 418.9828872724339*n - (xs*np.sin(np.sqrt(np.abs(xs)))).sum()

def rosenbrock(xs):
    n = xs.size
    return ((1-xs[:-1])**2 + 100*(xs[1:] - xs[:-1]**2)**2).sum()
