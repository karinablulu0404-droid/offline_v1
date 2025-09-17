import numpy as np
import sympy

# (2)定义并打印向量a = [2,1]。（10分）
a = np.array([2, 1])
print(a)

# (1)python定义函数f。（10分）
w0, w1, x = sympy.symbols("w0,w1,x")


def f(x):
    return w0 + w1 * x


# (2)计算f的导数（10分）
print(sympy.diff(f(x), x))


# (3)Python定义f的导函数d_f。（10分）
def d_f(x):
    return sympy.diff(f(x), x)


print(d_f(x))