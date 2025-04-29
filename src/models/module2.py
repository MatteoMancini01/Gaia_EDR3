import jax 
import jax.numpy as jnp
from math import factorial


def u_vec(alpha, delta):
    x_comp = jnp.cos(alpha)*jnp.cos(delta)
    y_comp = jnp.sin(alpha)*jnp.cos(delta)
    z_comp = jnp.sin(delta)
    return jnp.array(x_comp, y_comp, z_comp)

# Defining Legendre functions for VSH
def P_l0(x, l):

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
P_l = jax.jit(P_l0, static_argnames=["l"])

def P_lm0(x, l, m):
    """
    Computes the associated Legendre function P_{lm}(x)
    using your P_l(x, l) function.
    
    Args:
        x : array or scalar
        l : degree (integer)
        m : order (integer)
    
    Returns:
        P_lm(x)

    """
    if m>=0:

        # Define scalar function
        f_scalar = lambda x_val: P_l(x_val, l)

        # Take m-th derivative (scalar)
        for _ in range(m):
            f_scalar = jax.grad(f_scalar)

        if jnp.ndim(x) == 0:  # scalar input
            derivative_m = f_scalar(x)
        else:  # array input
            f_vector = jax.vmap(f_scalar)
            derivative_m = f_vector(x)

        prefactor = (1 - x**2)**(m/2)

        return prefactor*derivative_m

    else:
        # Define scalar function
        f_scalar = lambda x_val: P_l(x_val, l)

        # Take m-th derivative (scalar)
        for _ in range(abs(m)):
            f_scalar = jax.grad(f_scalar)

        if jnp.ndim(x) == 0:  # scalar input
            derivative_m = f_scalar(x)
        else:  # array input
            f_vector = jax.vmap(f_scalar)
            derivative_m = f_vector(x)
        # for both prefactors set -m as m in this case is negative and sign changes
        prefactor1 = (1 - x**2)**(-m/2)
        prefactor2 = (-1)**(-m)*factorial(l+m)/factorial(l-m)

        return prefactor2*prefactor1*derivative_m
P_lm = jax.jit(P_lm0, static_argnames=["l","m"])

def Y_lm0(alpha, delta, l, m):

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
Y_lm = jax.jit(Y_lm0, static_argnames=["l","m"])

def Y_slm0(alpha, delta, l, m):

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
Y_slm = jax.jit(Y_slm0, static_argnames=["l","m"])



def T_lm_scalar0(alpha, delta, l, m):

    """
    Function designed for the torodoidal function
    """

    e_alpha = jnp.array([
        -jnp.sin(alpha) / jnp.cos(delta),
         jnp.cos(alpha) / jnp.cos(delta),
         0.0
    ])

    e_delta = jnp.array([
        -jnp.cos(alpha) * jnp.sin(delta),
        -jnp.sin(alpha) * jnp.sin(delta),
         jnp.cos(delta)
    ])

    prefactor = 1 / jnp.sqrt(l * (l + 1))

    grad_real_alpha = jax.grad(lambda a: jnp.real(Y_lm(a, delta, l, m)))
    grad_imag_alpha = jax.grad(lambda a: jnp.imag(Y_lm(a, delta, l, m)))
    grad_real_delta = jax.grad(lambda d: jnp.real(Y_lm(alpha, d, l, m)))
    grad_imag_delta = jax.grad(lambda d: jnp.imag(Y_lm(alpha, d, l, m)))

    Ylm_grad_alpha = grad_real_alpha(alpha) + 1j * grad_imag_alpha(alpha)
    Ylm_grad_delta = grad_real_delta(delta) + 1j * grad_imag_delta(delta)

    return prefactor * (Ylm_grad_delta*e_alpha - (1/jnp.cos(delta))*Ylm_grad_alpha*e_delta)
T_lm_scalar = jax.jit(T_lm_scalar0, static_argnames=["l","m"])


def T_slm_scalar0(alpha, delta, l, m):
    """
    Function designed to take the complex conjugate of S_lm
    """
    return jnp.conj(T_lm_scalar(alpha, delta, l, m))
T_slm_scalar = jax.jit(T_slm_scalar0, static_argnames=["l","m"])


def T_lm0(alpha, delta, l, m):
    alpha = jnp.atleast_1d(alpha)
    delta = jnp.atleast_1d(delta)

    T_fn = jax.vmap(lambda a, d: T_lm_scalar(a, d, l, m))

    result = T_fn(alpha, delta)

    # If inputs were scalars, unwrap the result
    return result[0] if result.shape[0] == 1 else result
T_lm = jax.jit(T_lm0, static_argnames=["l","m"])

def T_slm0(alpha, delta, l, m):
    alpha = jnp.atleast_1d(alpha)
    delta = jnp.atleast_1d(delta)

    T_fn = jax.vmap(lambda a, d: T_slm_scalar(a, d, l, m))

    result = T_fn(alpha, delta)

    # If inputs were scalars, unwrap the result
    return result[0] if result.shape[0] == 1 else result
T_slm = jax.jit(T_slm0, static_argnames=["l","m"])

def S_lm_scalar0(alpha, delta, l, m):
    """
    Function designed for the spheroidal function
    """

    e_alpha = jnp.array([
        -jnp.sin(alpha) / jnp.cos(delta),
         jnp.cos(alpha) / jnp.cos(delta),
         0.0
    ])

    e_delta = jnp.array([
        -jnp.cos(alpha) * jnp.sin(delta),
        -jnp.sin(alpha) * jnp.sin(delta),
         jnp.cos(delta)
    ])

    prefactor = 1 / jnp.sqrt(l * (l + 1))

    grad_real_alpha = jax.grad(lambda a: jnp.real(Y_lm(a, delta, l, m)))
    grad_imag_alpha = jax.grad(lambda a: jnp.imag(Y_lm(a, delta, l, m)))
    grad_real_delta = jax.grad(lambda d: jnp.real(Y_lm(alpha, d, l, m)))
    grad_imag_delta = jax.grad(lambda d: jnp.imag(Y_lm(alpha, d, l, m)))

    Ylm_grad_alpha = grad_real_alpha(alpha) + 1j * grad_imag_alpha(alpha)
    Ylm_grad_delta = grad_real_delta(delta) + 1j * grad_imag_delta(delta)

    return prefactor*((1/jnp.cos(delta))*Ylm_grad_alpha*e_alpha + Ylm_grad_delta*e_delta)
S_lm_scalar = jax.jit(S_lm_scalar0, static_argnames=["l","m"])

def S_slm_scalar0(alpha, delta, l, m):
    """
    Function designed to take the complex conjugate of S_lm
    """
    return jnp.conj(S_lm_scalar(alpha, delta, l, m))
S_slm_scalar = jax.jit(S_slm_scalar0, static_argnames=["l","m"])

def S_lm0(alpha, delta, l, m):
    alpha = jnp.atleast_1d(alpha)
    delta = jnp.atleast_1d(delta)

    T_fn = jax.vmap(lambda a, d: S_lm_scalar(a, d, l, m))

    result = T_fn(alpha, delta)

    # If inputs were scalars, unwrap the result
    return result[0] if result.shape[0] == 1 else result
S_lm = jax.jit(S_lm0, static_argnames=["l","m"])

def S_slm0(alpha, delta, l, m):
    alpha = jnp.atleast_1d(alpha)
    delta = jnp.atleast_1d(delta)

    T_fn = jax.vmap(lambda a, d: S_slm_scalar(a, d, l, m))

    result = T_fn(alpha, delta)

    # If inputs were scalars, unwrap the result
    return result[0] if result.shape[0] == 1 else result
S_slm = jax.jit(S_slm0, static_argnames=["l","m"])