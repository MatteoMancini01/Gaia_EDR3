#%%
import jax 
import jax.numpy as jnp
from jax.scipy.special import gamma, factorial
import timeit
import math
import matplotlib.pyplot as plt
import numpy as np
#%%

print(gamma(3+1))
print(jnp.prod(jnp.arange(1,3+1)))
print(factorial(4))
#%%
# Need to determine which method is the best one, 
# This will be assessed in terms of run time and accuracy

def factorial_jax(n):
    return jnp.prod(jnp.arange(1,n+1, dtype=jnp.float64))
def factorial_jax_scipy_gamma(n):
    return gamma(n+1)
def factorial_jax_scipy_f(n):
    return factorial(n)
def factorial_jax_stable(n):
    n = jnp.asarray(n)
    logs = jnp.log(jnp.arange(1, n+1, dtype=jnp.float64))
    return jnp.exp(jnp.sum(logs))

n = 100
steps = 1000
true_factorial = math.factorial(n)
fj0 = factorial_jax(n)
fj1 = factorial_jax_scipy_gamma(n)
fj2 = factorial_jax_scipy_f(n)
fj_stable = factorial_jax_stable(n)

time_jax_fact = timeit.timeit(lambda:factorial_jax(n), number = steps)
time_jax_scipy1 = timeit.timeit(lambda:factorial_jax_scipy_gamma(n), number = steps)
time_jax_scipy2 = timeit.timeit(lambda:factorial_jax_scipy_f(n), number=steps)
time_stable_fact = timeit.timeit(lambda:factorial_jax_stable(n), number=steps)
time_math_factorial = timeit.timeit(lambda:math.factorial(n), number=steps)


print(f"Time required to compute factorial of {n} using pure jax (for {steps} steps)",time_jax_fact)
print(f"Time required to compute factorial of {n} using jax.scipy.special.gamma (for {steps} steps)",time_jax_scipy1)
print(f"Time required to compute factorial of {n} using jax.scipy.special.factorial (for {steps} steps)",time_jax_scipy2)
print(f"Time required to compute factorial of {n} using factorial_jax_stable (for {steps} steps)",time_stable_fact)
print(time_math_factorial)
print("")
print("Errors:")
#print("Absolute error factorial prure jax", jnp.abs(true_factorial-fj0))
#print("Absolute error factorial prure jax", jnp.abs(true_factorial-fj1))

#%%
print(true_factorial)
print(fj0)
print(fj1)
print(fj2)
print(fj_stable)
print(factorial_jax(10))
print(type(fj1))
print(type(true_factorial))
#%%
print(factorial_jax_stable(100))
#%%
from src.models.module2 import*
#%%
P_1 = P_l(jnp.sin(jnp.pi/2),1)
print(P_1)

#%%
print(P_l())

#%%
delta = jnp.linspace(0, 2*jnp.pi, num=500)

plt.plot(delta, P_l(jnp.sin(delta),1), label='l=1')
plt.plot(delta, P_l(jnp.sin(delta),2), label='l=2')
plt.plot(delta, P_l(jnp.sin(delta),3), label='l=3')
plt.plot(delta, P_l(jnp.sin(delta),4), label='l=4')
plt.plot(delta, P_l(jnp.sin(delta),5), label='l=5')
plt.title(f'Testing $P_l$ function')
plt.xlabel(f'$\delta$')
plt.ylabel(f'$P_l(sin(\delta))$')
plt.legend()
#%%
print(P_l(jnp.array([1,2]),2))
#%%
delta = jnp.linspace(0, 2*jnp.pi, num=500)

plt.plot(delta, P_l(jnp.sin(delta),3), label='l=3')
plt.plot(delta, P_lm(jnp.sin(delta),3,2), label='l=3, m=2')
plt.title(f'Testing $P_l$ function')
plt.xlabel(f'$\delta$')
plt.ylabel(f'$P_l(sin(\delta))$')
plt.legend()
#%%
P_lm(2, 1, 2)
#%%



@partial(jit, static_argnames=('l', 'm'))
def P_lm0(x, l, m):
    """
    Computes the associated Legendre function P_{lm}(x),
    with proper support for scalar, 1D, and 2D array inputs.
    """
    # Define the scalar base Legendre polynomial
    def P_l_scalar(x_val):
        sum0 = 0.
        for k in range(0, l//2 + 1):
            term = (-1)**k*factorial(2*l - 2*k)/(
                2**l*factorial(k)*factorial(l - k)*factorial(l - 2*k))*x_val**(l - 2 * k)
            sum0 += term
        return sum0

    # Differentiate m times
    for _ in range(abs(m)):
        P_l_scalar = jax.grad(P_l_scalar)

    def P_lm_scalar(x_val):
        base = P_l_scalar(x_val)
        if m >= 0:
            return (1 - x_val**2)**(m/2)*base
        else:
            prefactor = (-1)**(-m)*factorial(l + m)/factorial(l - m)
            return prefactor*(1 - x_val**2)**(-m/2)*base

    # Dispatch on input shape
    if jnp.ndim(x) == 0:
        return P_lm_scalar(x)
    elif jnp.ndim(x) == 1:
        return jax.vmap(P_lm_scalar)(x)
    elif jnp.ndim(x) == 2:
        return jax.vmap(jax.vmap(P_lm_scalar))(x)
    else:
        raise ValueError("Unsupported input dimension for P_lm")

P_lm = jax.jit(P_lm0, static_argnames=["l","m"])

#%%

@partial(jit, static_argnames=('l', 'm'))
def T_lm_scalar0(alpha, delta, l, m):

    """
    Function designed for the torodoidal function
    """
    e_alpha, e_delta = basis_vectors(alpha,delta)

    prefactor = 1 / jnp.sqrt(l*(l + 1))

    grad_real_alpha = jax.grad(lambda a: jnp.real(Y_lm(a, delta, l, m)))
    grad_imag_alpha = jax.grad(lambda a: jnp.imag(Y_lm(a, delta, l, m)))
    grad_real_delta = jax.grad(lambda d: jnp.real(Y_lm(alpha, d, l, m)))
    grad_imag_delta = jax.grad(lambda d: jnp.imag(Y_lm(alpha, d, l, m)))

    Ylm_grad_alpha = grad_real_alpha(alpha) + 1j * grad_imag_alpha(alpha)
    Ylm_grad_delta = grad_real_delta(delta) + 1j * grad_imag_delta(delta)

    return prefactor*(Ylm_grad_delta*e_alpha - (1/jnp.cos(delta))*Ylm_grad_alpha*e_delta)
T_lm_scalar = jax.jit(T_lm_scalar0, static_argnames=["l","m"])

#%%
import timeit
import math
import matplotlib.pyplot as plt
import numpy as np
from functools import partial, lru_cache
import jax
import jax.numpy as jnp
#%%
@lru_cache(maxsize=256)
def factorial(n: int):
    return math.factorial(n)

@lru_cache(maxsize=256)
def make_legendre_polynomial(l: int):
    def P_l_scalar(x_val):
        sum0 = 0.
        for k in range(0, l//2 + 1):
            term = (-1)**k*factorial(2*l - 2*k)/(
                2**l*factorial(k)*factorial(l - k)*factorial(l - 2*k)
            ) * x_val**(l - 2*k)
            sum0 += term
        return sum0
    return P_l_scalar

def make_P_lm_scalar(l, m):
    P_l = make_legendre_polynomial(l)
    for _ in range(abs(m)):
        P_l = jax.grad(P_l)

    def P_lm(x_val):
        base = P_l(x_val)
        if m >= 0:
            return (1 - x_val**2)**(m/2)*base
        else:
            prefactor = (-1)**(-m)*factorial(l + m)/factorial(l - m)
            return prefactor * (1 - x_val**2)**(-m/2)*base
    return jax.jit(P_lm)

def P_lm(x, l, m):
    P_lm_scalar = make_P_lm_scalar(l, m)

    if jnp.ndim(x) == 0:
        return P_lm_scalar(x)
    elif jnp.ndim(x) == 1:
        return jax.vmap(P_lm_scalar)(x)
    elif jnp.ndim(x) == 2:
        return jax.vmap(jax.vmap(P_lm_scalar))(x)
    else:
        raise ValueError("Unsupported input dimension for P_lm")
P_lm_wrapped = jax.jit(P_lm, static_argnames=["l", "m"])
#%%
result = timeit.timeit(lambda: P_lm_wrapped(50., 3, 2), number=1000)
print(result)
#%%

from src.models.module3 import*
import matplotlib.pyplot as plt
import jax.numpy as jnp
#%%
fig = plt.figure()
ax  = fig.add_subplot(projection='3d')
l, m = 2, 1
alpha = jnp.linspace(jnp.pi, 2*jnp.pi, num=500)
delta = jnp.linspace(jnp.pi, 2*jnp.pi, num=500)
a,d = jnp.meshgrid(alpha, delta)
Y_lm_fun = Y_lm(a, d, l, m)

ax.plot_surface(a, d, Y_lm_fun.real, label = 'Real Part')
ax.plot_surface(a, d, Y_lm_fun.imag, label = 'Imaginary Part')
ax.set_title("Plotting Example")
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\delta$')
ax.set_zlabel(f'Y_{l}{m}')
plt.legend()
plt.tight_layout()
plt.show()
#%%
fig = plt.figure()
ax  = fig.add_subplot(projection='3d')
l, m = 2, 1
alpha = jnp.linspace(jnp.pi, 2*jnp.pi, num=500)
delta = jnp.linspace(jnp.pi, 2*jnp.pi, num=500)
a,d = jnp.meshgrid(alpha, delta)
Y_lm_fun = Y_slm(a, d, l, m)

ax.plot_surface(a, d, Y_lm_fun.real, label = 'Real Part')
ax.plot_surface(a, d, Y_lm_fun.imag, label = 'Imaginary Part')
ax.set_title("Plotting Example")
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\delta$')
ax.set_zlabel(f'Y_{l}{m}')
plt.legend()
plt.tight_layout()
plt.show()
#%%
l,m = 2, 2
f1 = Y_lm(0.1, 2.37, l, m)
print('Real part of f1 = ', f1.real)
print('Imaginary part of f1 = ', f1.imag)
print('Complex form of f1 = ', f1)
print('Complex conjugate of f1, f1* = ', jnp.conjugate(f1))
#%%
from src.models.module3 import*
import matplotlib.pyplot as plt
import jax.numpy as jnp
#%%
result = T_lm_scalar(jnp.pi/7, jnp.pi/3, 10, 2)
print(f'Real part {result.real}')
print(f'Imaginary part {result.imag}')
print(f'Complex vector: {result}')

# Example of dot product
dot_result = result.T @ jnp.conj(result)
print(f'Dot product of result with its complex conjugate {dot_result}')

# Constructing a Matrix
v = T_lm_scalar(jnp.pi, jnp.pi/3, 1, 0)
u = T_lm_scalar(3*jnp.pi, jnp.pi/5, 1, 1)

M = jnp.vstack([v,u])
print(f'Constructed matrix M = {M}')
#%%
alpha0, delta0 = jnp.array([0.2, 1.4, -6.4]), jnp.array([-5.236, 8.75, 5.8])
alpha1, delta1 = jnp.array([-1.335, 9.21, 8.2]), jnp.array([-5.2, -7.42, 0.2356])
v = T_lm(alpha0, delta0, 4, 0, grid = False)
u = T_lm(alpha1, delta1, 6, 2, grid = False)

v_u = v @ u
u_v = u @ v

print(f'vT.u = {v_u}')
print(f'uT.v = {u_v}')
#%%

import matplotlib.pyplot as plt
import numpy as np
from src.models.vsh_model import*
#%%

x = np.linspace(-1, 1, 100)
plt.plot(x, make_legendre_polynomial(0, x))