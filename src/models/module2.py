import jax 
import jax.numpy as jnp
import math
from jax import jit
from functools import partial, lru_cache

@lru_cache(maxsize=256)
def factorial(n: int):
    return math.factorial(n)

def u_vec(alpha, delta):
    x_comp = jnp.cos(alpha)*jnp.cos(delta)
    y_comp = jnp.sin(alpha)*jnp.cos(delta)
    z_comp = jnp.sin(delta)
    return jnp.array(x_comp, y_comp, z_comp)

# Defining Legendre functions for VSH
def P_l(x, l):

    """
    Computes the associated Legendre function P_{l}(x),
    using the binomial expansion.

    Args:
        x : array or scalar
        l : degree (integer)

    Returns:
        P+l(x)

    """

    sum0 = jnp.zeros_like(x)
    for k in range(0,l//2+1):
        sum0 += (-1)**k*(factorial(2*l - 2*k))/(2**l*
                        factorial(k)*factorial(l-k)*factorial(l-2*k))*x**(l-2*k)
        
    return sum0
P_l = jax.jit(P_l, static_argnames=["l"])

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
P_lm = jax.jit(P_lm, static_argnames=["l", "m"])


@partial(jit, static_argnames=('l', 'm'))
def Y_lm(alpha, delta, l, m):

    """
    Args:
        delta : array or scalar
        alpha : array or scalar
        l : degree (integer)
        m : order (integer)
    
    Returns:
        Y_lm(alpha, delta)
    """

    norm = (-1)**m*jnp.sqrt(((2*l + 1)/(4*jnp.pi))*(factorial(l - m)/factorial(l + m)))
    P = P_lm(jnp.sin(delta), l, m)

    exp = jnp.exp(1j*m*alpha)

    return norm*P*exp
Y_lm = jax.jit(Y_lm, static_argnames=["l","m"])

@partial(jit, static_argnames=('l', 'm'))
def Y_slm(alpha, delta, l, m):

    """
    Args:
        delta : array or scalar
        alpha : array or scalar
        l : degree (integer)
        m : order (integer)
    
    Returns:
        Y_lm(alpha, delta)
    """

    norm = (-1)**m*jnp.sqrt(((2*l + 1)/(4*jnp.pi))*(factorial(l - m)/factorial(l + m)))
    P = P_lm(jnp.sin(delta), l, m)

    exp = jnp.exp(-1j*m*alpha)

    return norm*P*exp
Y_slm = jax.jit(Y_slm, static_argnames=["l","m"])


def basis_vectors(alpha, delta):
    e_alpha = jnp.stack([-jnp.sin(alpha), jnp.cos(alpha), 0.0], axis=0)
    e_delta = jnp.stack([-jnp.cos(alpha) * jnp.sin(delta),
                         -jnp.sin(alpha) * jnp.sin(delta),
                          jnp.cos(delta)], axis=0)
    return e_alpha, e_delta


@partial(jit, static_argnames=('l', 'm'))
def T_lm_scalar(alpha, delta, l, m):

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
T_lm_scalar = jax.jit(T_lm_scalar, static_argnames=["l","m"])

@partial(jit, static_argnames=('l', 'm'))
def T_slm_scalar(alpha, delta, l, m):
    """
    Function designed to take the complex conjugate of S_lm
    """
    return jnp.conj(T_lm_scalar(alpha, delta, l, m))
T_slm_scalar = jax.jit(T_slm_scalar, static_argnames=["l","m"])

@partial(jit, static_argnames=('l', 'm', 'grid'))
def T_lm(alpha, delta, l, m, grid=True):
    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return T_lm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: T_lm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: T_lm_scalar(a, d, l, m))(alpha, delta)

T_lm = jax.jit(T_lm, static_argnames=["l", "m", "grid"])

@partial(jit, static_argnames=('l', 'm', 'grid'))
def T_slm(alpha, delta, l, m, grid=True):
    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return T_slm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: T_slm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: T_slm_scalar(a, d, l, m))(alpha, delta)

T_slm = jax.jit(T_slm, static_argnames=["l", "m", "grid"])

@partial(jit, static_argnames=('l', 'm'))
def S_lm_scalar(alpha, delta, l, m):
    """
    Function designed for the spheroidal function
    """

    e_alpha, e_delta = basis_vectors(alpha,delta)

    prefactor = 1/jnp.sqrt(l*(l + 1))

    grad_real_alpha = jax.grad(lambda a: jnp.real(Y_lm(a, delta, l, m)))
    grad_imag_alpha = jax.grad(lambda a: jnp.imag(Y_lm(a, delta, l, m)))
    grad_real_delta = jax.grad(lambda d: jnp.real(Y_lm(alpha, d, l, m)))
    grad_imag_delta = jax.grad(lambda d: jnp.imag(Y_lm(alpha, d, l, m)))

    Ylm_grad_alpha = grad_real_alpha(alpha) + 1j * grad_imag_alpha(alpha)
    Ylm_grad_delta = grad_real_delta(delta) + 1j * grad_imag_delta(delta)

    return prefactor*((1/jnp.cos(delta))*Ylm_grad_alpha*e_alpha + Ylm_grad_delta*e_delta)
S_lm_scalar = jax.jit(S_lm_scalar, static_argnames=["l","m"])

@partial(jit, static_argnames=('l', 'm'))
def S_slm_scalar(alpha, delta, l, m):
    """
    Function designed to take the complex conjugate of S_lm
    """
    return jnp.conj(S_lm_scalar(alpha, delta, l, m))
S_slm_scalar = jax.jit(S_slm_scalar, static_argnames=["l","m"])

@partial(jit, static_argnames=('l', 'm', 'grid'))
def S_lm(alpha, delta, l, m, grid=True):
    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return S_lm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: S_lm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: S_lm_scalar(a, d, l, m))(alpha, delta)

S_lm = jax.jit(S_lm, static_argnames=["l", "m", "grid"])

@partial(jit, static_argnames=('l', 'm', 'grid'))
def S_slm(alpha, delta, l, m, grid=True):
    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return S_slm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: S_slm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: S_slm_scalar(a, d, l, m))(alpha, delta)

S_slm = jax.jit(S_slm, static_argnames=["l", "m", "grid"])