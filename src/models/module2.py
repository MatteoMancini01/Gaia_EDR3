import jax 
import jax.numpy as jnp
from jax import jit 
from math import factorial

@jit
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

def P_lm(x, l, m):

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

    return prefactor * derivative_m

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
    P_lm_fun = P_lm(jnp.sin(delta), l, m)

    imag_part = norm*P_lm_fun*jnp.sin(m*alpha)
    real_part = norm*P_lm_fun*jnp.cos(m*alpha)

    return real_part, imag_part

def Y_slm(alpha, delta, l, m):
    Y_lm_ = Y_lm(alpha, delta, l, m)[0]
    real_conj = (-1)**m*Y_lm_[0]
    imag_conj = (-1)**(m + 1)*Y_lm_[1]

    return real_conj, imag_conj
