from math import sin, cos, exp, sqrt, pi

import dlib
from tuning.metaheuristic.function_utils import *

def fitness(*args):
    # print(args)

    res = 0
    for a in args:
        res += a ** 2
    return res

def holder_table(x0,x1):
    return -abs(sin(x0)*cos(x1)*exp(abs(1-sqrt(x0*x0+x1*x1)/pi)))

x, y = dlib.find_min_global(
    fitness,
    [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10],  # Lower bound constraints on x0 and x1 respectively
    [ 10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10],  # Upper bound constraints on x0 and x1 respectively
    10
)

print("++++++++++++++++++++++++++++++=")
print(y)
print(x)