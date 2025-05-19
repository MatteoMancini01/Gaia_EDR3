import jax 
import jax.numpy as jnp
import math
from jax import jit, vmap
from functools import partial, lru_cache

@lru_cache(maxsize=256)
def factorial(n: int):
    return math.factorial(n)

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


@jit
def basis_vectors(alpha, delta):
    e_alpha = jnp.stack([-jnp.sin(alpha), jnp.cos(alpha), 0.0], axis=0)
    e_delta = jnp.stack([-jnp.cos(alpha) * jnp.sin(delta),
                         -jnp.sin(alpha) * jnp.sin(delta),
                          jnp.cos(delta)], axis=0)
    return e_alpha, e_delta

@lru_cache(maxsize=256)
def make_Y_lm_gradients(l:int, m:int):
    def grad_alpha(alpha, delta):
        real_part = jax.grad(lambda a: jnp.real(Y_lm(a, delta, l, m)))(alpha)
        imag_part = jax.grad(lambda a: jnp.imag(Y_lm(a, delta, l, m)))(alpha)
        return real_part + 1j * imag_part

    def grad_delta(alpha, delta):
        real_part = jax.grad(lambda d: jnp.real(Y_lm(alpha, d, l, m)))(delta)
        imag_part = jax.grad(lambda d: jnp.imag(Y_lm(alpha, d, l, m)))(delta)
        return real_part + 1j*imag_part

    return grad_alpha, grad_delta

@partial(jit, static_argnames=('l', 'm'))
def T_lm_scalar(alpha, delta, l, m):

    """
    Function designed for the torodoidal function
    """
    e_alpha, e_delta = basis_vectors(alpha,delta)

    prefactor = 1 / jnp.sqrt(l*(l + 1))

    grad_alpha, grad_delta = make_Y_lm_gradients(l, m)
    Ylm_grad_alpha = grad_alpha(alpha, delta)
    Ylm_grad_delta = grad_delta(alpha, delta)
    safe_cos = jnp.where(jnp.abs(jnp.cos(delta)) < 1e-6, 1e-6, jnp.cos(delta))

    return prefactor*(Ylm_grad_delta*e_alpha - (1/safe_cos)*Ylm_grad_alpha*e_delta)


@partial(jit, static_argnames=('l', 'm'))
def T_slm_scalar(alpha, delta, l, m):
    """
    Function designed to take the complex conjugate of S_lm
    """
    return jnp.conj(T_lm_scalar(alpha, delta, l, m))


@partial(jit, static_argnames=('l', 'm', 'grid'))
def T_lm(alpha, delta, l, m, grid=True):
    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return T_lm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: T_lm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: T_lm_scalar(a, d, l, m))(alpha, delta)



@partial(jit, static_argnames=('l', 'm', 'grid'))
def T_slm(alpha, delta, l, m, grid=True):
    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return T_slm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: T_slm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: T_slm_scalar(a, d, l, m))(alpha, delta)



@partial(jit, static_argnames=('l', 'm'))
def S_lm_scalar(alpha, delta, l, m):
    """
    Function designed for the spheroidal function
    """

    e_alpha, e_delta = basis_vectors(alpha,delta)

    prefactor = 1/jnp.sqrt(l*(l + 1))

    grad_alpha, grad_delta = make_Y_lm_gradients(l, m)
    Ylm_grad_alpha = grad_alpha(alpha, delta)
    Ylm_grad_delta = grad_delta(alpha, delta)
    safe_cos = jnp.where(jnp.abs(jnp.cos(delta)) < 1e-6, 1e-6, jnp.cos(delta))

    return prefactor*((1/safe_cos)*Ylm_grad_alpha*e_alpha + Ylm_grad_delta*e_delta)


@partial(jit, static_argnames=('l', 'm'))
def S_slm_scalar(alpha, delta, l, m):
    """
    Function designed to take the complex conjugate of S_lm
    """
    return jnp.conj(S_lm_scalar(alpha, delta, l, m))


@partial(jit, static_argnames=('l', 'm', 'grid'))
def S_lm(alpha, delta, l, m, grid=True):
    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return S_lm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: S_lm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: S_lm_scalar(a, d, l, m))(alpha, delta)



@partial(jit, static_argnames=('l', 'm', 'grid'))
def S_slm(alpha, delta, l, m, grid=True):
    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return S_slm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: S_slm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: S_slm_scalar(a, d, l, m))(alpha, delta)



@partial(jit, static_argnames=('grid'))
def toy_model_l_1(alpha, delta, theta, grid):
    # theta = [t10, t11_real, t11_imag, s10, s11_real, s11_imag]
    # Compute vector spherical harmonics at this location
    a = alpha
    d = delta
    l=1
    V = (theta[0]*T_lm(a, d, l, 0, grid=grid) + 
         theta[1]*jnp.real(T_lm(a,d,l,1,grid=grid)) +
         theta[2]*jnp.imag(T_lm(a,d,l,1,grid=grid)) + 
         theta[3]*S_lm(a,d,l,0,grid=grid) + 
         theta[4]*jnp.real(S_lm(a,d,l,1,grid=grid)) + 
         theta[5]*jnp.imag(S_lm(a,d,l,1,grid=grid)) 
         ) 
    return V

@jax.jit
def toy_least_square(angles, obs, error, t_10, t_11r, t_11i, s_10, s_11r, s_11i):
    """
    angles: observed angles (ra, dec)
    error: uncertainites on proper motion (sigma_mu_alpha*, sigma_mu_delta)
    theta: parameters (e.g. t_10, s_10, etc.)
    obs: EDR3 Gaia observed proper motion (mu_alpha_obs, mu_delta_obs)
    """
    alpha, delta = angles
    mu_a_obs, mu_d_obs = obs
    s_mu_a, s_mu_d, rho = error
    theta = jnp.array([t_10, t_11r, t_11i, s_10, s_11r, s_11i])

    def per_point(alpha_i, delta_i, mu_a_i, mu_d_i, s_a, s_d, r):
        e_a, e_d = basis_vectors(alpha_i, delta_i)

        A = jnp.array([
            [s_a**2, r * s_a * s_d],
            [r * s_a * s_d, s_d**2]
        ])

        V = toy_model_l_1(alpha_i, delta_i, theta, False)
        V_alpha = jnp.vdot(V, e_a).real
        V_delta = jnp.vdot(V, e_d).real

        D = jnp.array([mu_a_i - V_alpha, mu_d_i - V_delta])
        x = jnp.linalg.solve(A, D)
        return D @ x

    batched_fn = jax.vmap(per_point)
    losses = batched_fn(alpha, delta, mu_a_obs, mu_d_obs, s_mu_a, s_mu_d, rho)
    return jnp.sum(losses)

@partial(jit, static_argnames=['lmax', 'grid'])
def model_vsh(alpha, delta, theta, lmax, grid):
    """
    General VSH model up to lmax.
    theta: flattened array of all [t_lm_real/imag, s_lm_real/imag] for l=1..lmax
    """

    a = alpha
    d = delta
    V = jnp.zeros(3, dtype=jnp.complex64)

    index = 0
    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            T = T_lm(a, d, l, m, grid=grid)
            S = S_lm(a, d, l, m, grid=grid)

            if m == 0:
                t_lm = theta[index]
                s_lm = theta[index + 1]
                index += 2
                V += t_lm * T
                V += s_lm * S
            else:
                t_r, t_i = theta[index], theta[index + 1]
                s_r, s_i = theta[index + 2], theta[index + 3]
                index += 4

                V += t_r*jnp.real(T) + t_i*jnp.imag(T)
                V += s_r*jnp.real(S) + s_i*jnp.imag(S)

    return V

def count_vsh_coeffs(lmax):
    return 2*lmax*(lmax + 2)

@partial(jit, static_argnames=['lmax', 'grid'])
def least_square(angles, obs, error, theta, lmax, grid):
    alpha, delta = angles
    mu_a_obs, mu_d_obs = obs
    s_mu_a, s_mu_d, rho = error

    def per_point(alpha_i, delta_i, mu_a_i, mu_d_i, s_a, s_d, r):
        e_a, e_d = basis_vectors(alpha_i, delta_i)

        A = jnp.array([
            [s_a**2, r*s_a*s_d],
            [r*s_a*s_d, s_d**2]
        ])

        V = model_vsh(alpha_i, delta_i, theta, lmax=lmax, grid=grid)
        V_alpha = jnp.vdot(V, e_a).real
        V_delta = jnp.vdot(V, e_d).real

        D = jnp.array([mu_a_i - V_alpha, mu_d_i - V_delta])
        x = jnp.linalg.solve(A, D)
        return D @ x

    batched_fn = vmap(per_point)
    losses = batched_fn(alpha, delta, mu_a_obs, mu_d_obs, s_mu_a, s_mu_d, rho)
    return jnp.sum(losses)

def spheroidal_vector_summary(s10, s11r, s11i):
    # Normalisation constants from VSH paper
    C0 = jnp.sqrt(8 * jnp.pi / 3)
    C1 = jnp.sqrt(4 * jnp.pi / 3)

    # Convert s_lm to vector components
    G3 = s10 / C0
    G1 = -s11r / C1
    G2 = -s11i / C1

    # Magnitude
    G_mag = jnp.sqrt(G1**2 + G2**2 + G3**2)

    # Direction (equatorial)
    dec = jnp.arcsin(G3/G_mag)
    ra = jnp.arctan2(G2, G1) % (2*jnp.pi)

    # Convert to μas/yr and degrees for readability
    G_mag_uas = G_mag*1000  # mas/yr -> μas/yr
    ra_deg = jnp.rad2deg(ra)
    dec_deg = jnp.rad2deg(dec)

    return {
        "G_vector (mas/yr)": jnp.array([G1, G2, G3]),
        "Magnitude (μas/yr)": G_mag_uas,
        "RA (deg)": ra_deg,
        "Dec (deg)": dec_deg
    }

def toroidal_vector_summary(t10, t11r, t11i):
    # Normalisation constants from VSH paper
    C0 = jnp.sqrt(8 * jnp.pi / 3)
    C1 = jnp.sqrt(4 * jnp.pi / 3)

    # Convert t_lm to vector components
    R3 = t10/C0
    R1 = -t11r/C1
    R2 = -t11i/C1

    # Magnitude
    R_mag = jnp.sqrt(R1**2 + R2**2 + R3**2)

    # Direction (equatorial)
    dec = jnp.arcsin(R3/R_mag)
    ra = jnp.arctan2(R2, R1) % (2*jnp.pi)

    # Convert to μas/yr and degrees
    R_mag_uas = R_mag*1000  # mas/yr -> μas/yr
    ra_deg = jnp.rad2deg(ra)
    dec_deg = jnp.rad2deg(dec)

    return {
        "R_vector (mas/yr)": jnp.array([R1, R2, R3]),
        "Magnitude (μas/yr)": R_mag_uas,
        "RA (deg)": ra_deg,
        "Dec (deg)": dec_deg
    }

def vsh_minuit_limits(lmax, t_bound=0.01, s_bound=0.01):
    """
    Generate a dictionary of parameter limits for Minuit based on lmax.
    Returns: dict of {parameter_name: (lower, upper)}
    """
    limits = {}
    idx = 0

    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            if m == 0:
                # t_lm, s_lm (1 real each)
                limits[f'x{idx}'] = (-t_bound, t_bound)
                limits[f'x{idx+1}'] = (-s_bound, s_bound)
                idx += 2
            else:
                # Re(t), Im(t), Re(s), Im(s)
                limits[f'x{idx}'] = (-t_bound, t_bound)
                limits[f'x{idx+1}'] = (-t_bound, t_bound)
                limits[f'x{idx+2}'] = (-s_bound, s_bound)
                limits[f'x{idx+3}'] = (-s_bound, s_bound)
                idx += 4

    return limits