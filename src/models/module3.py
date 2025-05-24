import jax.numpy as jnp
import jax 
from jax import jit, vmap
import math
from functools import partial, lru_cache
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

@lru_cache(maxsize=256)
def factorial(n: int):

    """
    Memoised factorial function using Python's built-in math.factorial.

    Args:
        n (int): Non-negative integer to compute the factorial of.

    Returns:
        int: The factorial of n (i.e., n!).

    Notes:
        - Caches up to 256 results to speed up repeated calls.
        - Used internally by various vector spherical harmonic (VSH) components.

    Example:
    >>> from src.models.vsh_model import factorial
    >>> print(factorial(5))
    """

    return math.factorial(n)

@lru_cache(maxsize=256)
def make_legendre_polynomial(l: int):

    """
    Generates a scalar function for the Legendre polynomial P_l(x) using a binomial series expansion.

    Args:
        l (int): Degree of the Legendre polynomial.

    Returns:
        Callable[[float or jnp.ndarray], float or jnp.ndarray]: A function P_l(x) that evaluates the Legendre polynomial of degree l at any scalar or array input x.

    Notes:
        - This is the unassociated Legendre polynomial P_l(x), used as a building block for associated functions.
        - Memoised with `lru_cache` to avoid recomputation.
    """

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

    """
    Returns a scalar function to evaluate the associated Legendre function P_lm(x).

    Args:
        l (int): Degree of the associated Legendre function.
        m (int): Order of the associated Legendre function (can be negative).

    Returns:
        Callable[[float or jnp.ndarray], float or jnp.ndarray]: A function P_lm(x) that evaluates the associated Legendre function P_lm(x).

    Notes:
        - Builds on `make_legendre_polynomial(l)` by applying `jax.grad` m times to obtain derivatives.
        - Handles both positive and negative m using the standard Condon–Shortley phase and normalisation.
        - Suitable for both scalar and batched inputs when wrapped in a vmapped evaluator.
    """

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

    """
    Evaluates the associated Legendre function P_lm(x) for scalars, vectors, or matrices of x.

    Args:
        x (float or jnp.ndarray): Input value(s) to evaluate P_lm at. Can be scalar, 1D, or 2D array.
        l (int): Degree of the function.
        m (int): Order of the function.

    Returns:
        float or jnp.ndarray: Value(s) of the associated Legendre function P_lm(x), matching the shape of input x.

    Raises:
        ValueError: If the input dimension of x is not supported.

    Notes:
        - Automatically applies vectorisation using `jax.vmap` for 1D and 2D inputs.
        - Uses `make_P_lm_scalar` internally for scalar-compatible evaluation.
        - JIT-compiled and static over (l, m) for performance.

    Example:
        >>> from src.models.vsh_model import P_lm
        >>> import jax.numpy as jnp
        >>> import matplotlib.pyplot as plt
        >>> # Plotting example
        >>> delta = jnp.linspace(0, 2*jnp.pi, num=500)
        >>> plt.plot(delta, P_lm(jnp.sin(delta),3,2), label='l=3, m=2')
        >>> plt.title(f'Testing $P_lm$ function')
        >>> plt.xlabel(f'$\delta$')
        >>> plt.ylabel(f'$P_lm(sin(\delta))$')
        >>> plt.legend()
        >>> plt.show()
        >>> # Scalar computation example
        >>> print(P_lm(2., 2, 1))

    """

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
    Computes the complex spherical harmonic function Y_lm(alpha, delta) evaluated at given celestial coordinates.

    Args:
        alpha (float or jnp.ndarray): Right ascension angle(s) in radians.
        delta (float or jnp.ndarray): Declination angle(s) in radians.
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.

    Returns:
        complex or jnp.ndarray: Value(s) of the spherical harmonic Y_lm at the input coordinates.

    Notes:
        - Returns the standard spherical harmonic with a +m exponential term: exp(+i.m.alpha).
        - Normalised according to the real-valued orthonormal convention over the sphere.
        - Internally uses `P_lm(sin(delta), l, m)` for the latitudinal dependence and complex exponential in alpha for longitudinal variation.
        - JIT-compiled and static over (l, m) for performance.
    
    Example:
        >>> from src.models.module3 import*
        >>> import matplotlib.pyplot as plt
        >>> import jax.numpy as jnp
        >>> # 3D Plot Example
        >>> fig = plt.figure()
        >>> ax  = fig.add_subplot(projection='3d')
        >>> l, m = 2, 1
        >>> alpha = jnp.linspace(jnp.pi, 2*jnp.pi, num=500)
        >>> delta = jnp.linspace(jnp.pi, 2*jnp.pi, num=500)
        >>> a,d = jnp.meshgrid(alpha, delta)
        >>> Y_lm_fun = Y_lm(a, d, l, m)
        >>> ax.plot_surface(a, d, Y_lm_fun.real, label = 'Real Part')
        >>> ax.plot_surface(a, d, Y_lm_fun.imag, label = 'Imaginary Part')
        >>> ax.set_title("Plotting Example")
        >>> ax.set_xlabel(r'$\alpha$')
        >>> ax.set_ylabel(r'$\delta$')
        >>> ax.set_zlabel(f'Y_{l}{m}')
        >>> plt.legend()
        >>> plt.tight_layout()
        >>> plt.show()
        >>> # Scalar Input example
        >>> f1 = Y_lm(0.1, 2.37, l, m)
        >>> print('Real part of f1 = ', f1.real)
        >>> print('Imaginary part of f1 = ', f1.imag)
        >>> print('Complex form of f1 = ', f1)
        >>> print('Complex conjugate of f1, f1* = ', jnp.conjugate(f1))

    """


    norm = (-1)**m*jnp.sqrt(((2*l + 1)/(4*jnp.pi))*(factorial(l - m)/factorial(l + m)))
    P = P_lm(jnp.sin(delta), l, m)

    exp = jnp.exp(1j*m*alpha)

    return norm*P*exp

@partial(jit, static_argnames=('l', 'm'))
def Y_slm(alpha, delta, l, m):

    """
    Computes the complex conjugate of the spherical harmonic Y_lm(alpha, delta), with a -m exponential phase.

    Args:
        alpha (float or jnp.ndarray): Right ascension angle(s) in radians.
        delta (float or jnp.ndarray): Declination angle(s) in radians.
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.

    Returns:
        complex or jnp.ndarray: Value(s) of the spherical harmonic conjugate Y_lm at the input coordinates.

    Notes:
        - Similar to `Y_lm`, but uses exp(-i.m.alpha) instead of exp(+i.m.alpha).
        - Often used when building conjugate harmonics for inverse transforms or adjoint operations (e.g., in VSH modeling).
        - Shares normalization and structure with `Y_lm`, but changes the sign of the azimuthal phase.
        - JIT-compiled and static over (l, m) for efficiency.
    
    Example:
        >>> from src.models.module3 import*
        >>> import matplotlib.pyplot as plt
        >>> import jax.numpy as jnp
        >>> # 3D Plot Example
        >>> fig = plt.figure()
        >>> ax  = fig.add_subplot(projection='3d')
        >>> l, m = 2, 1
        >>> alpha = jnp.linspace(jnp.pi, 2*jnp.pi, num=500)
        >>> delta = jnp.linspace(jnp.pi, 2*jnp.pi, num=500)
        >>> a,d = jnp.meshgrid(alpha, delta)
        >>> Y_slm_fun = Y_slm(a, d, l, m)
        >>> ax.plot_surface(a, d, Y_slm_fun.real, label = 'Real Part')
        >>> ax.plot_surface(a, d, Y_slm_fun.imag, label = 'Imaginary Part')
        >>> ax.set_title("Plotting Example")
        >>> ax.set_xlabel(r'$\alpha$')
        >>> ax.set_ylabel(r'$\delta$')
        >>> ax.set_zlabel(f'Y_{l}{m}')
        >>> plt.legend()
        >>> plt.tight_layout()
        >>> plt.show()
        >>> # Scalar Input example
        >>> f1 = Y_slm(0.1, 2.37, l, m)
        >>> print('Real part of f1 = ', f1.real)
        >>> print('Imaginary part of f1 = ', f1.imag)
        >>> print('Complex form of f1 = ', f1)
        >>> print('Complex conjugate of f1, f1* = ', jnp.conjugate(f1))
    """


    norm = (-1)**m*jnp.sqrt(((2*l + 1)/(4*jnp.pi))*(factorial(l - m)/factorial(l + m)))
    P = P_lm(jnp.sin(delta), l, m)

    exp = jnp.exp(-1j*m*alpha)

    return norm*P*exp

@jit
def basis_vectors(alpha, delta):

    """
    Computes the local unit basis vectors (e_alpha, e_delta) on the celestial sphere at given angular coordinates.

    Args:
        alpha (float): Right ascension in radians.
        delta (float): Declination in radians.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple of two vectors:
            - `e_alpha`: Unit vector in the increasing right ascension direction.
            - `e_delta`: Unit vector in the increasing declination direction.

    Notes:
        - Both vectors are expressed in 3D Cartesian coordinates.
        - These form an orthonormal tangent basis at the given point on the sphere.
        - Used in computing projections of vector fields (e.g., proper motions) and VSH gradients.
    
    Example:
        >>> from src.models.vsh_model import basis_vectors
        >>> import jax.numpy as jnp
        >>> alpha, delta = jnp.array([0., 1.2, -4.3]), jnp.array([-6., 2.4, 3.76])
        >>> e_alpha, e_delta = basis_vectors(alpha, delta)
        >>> print(f'given the alpha = {alpha}, the respective basis vector is e_alpha = {e_alpha}')
        >>> print(f'given the delta = {delta}, the respective basis vector is e_delta = {e_delta}')

    """

    e_alpha = jnp.stack([-jnp.sin(alpha), jnp.cos(alpha), 0.0], axis=0)
    e_delta = jnp.stack([-jnp.cos(alpha) * jnp.sin(delta),
                         -jnp.sin(alpha) * jnp.sin(delta),
                          jnp.cos(delta)], axis=0)
    return e_alpha, e_delta

@lru_cache(maxsize=256)
def make_Y_lm_gradients(l:int, m:int):

    """
    Constructs functions to compute the partial derivatives of the spherical harmonic Y_lm(alpha, delta) w.r.t. alpha and delta.

    Args:
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic.

    Returns:
        Tuple[Callable, Callable]:
            - `grad_alpha(alpha, delta)`: Computes dY_lm / dalpha at (alpha, delta), including both real and imaginary parts.
            - `grad_delta(alpha, delta)`: Computes dY_lm / ddelta at (alpha, delta), including both real and imaginary parts.

    Notes:
        - Derivatives are computed using JAX's automatic differentiation (`jax.grad`) on the real and imaginary parts.
        - Output is complex-valued: real and imaginary parts are differentiated separately and recombined.
        - The returned gradient functions are intended for use in computing vector spherical harmonics (e.g., toroidal/spheroidal modes).
        - This function is memorised with `lru_cache` to improve performance when repeatedly evaluating gradients for fixed (l, m).
    """

    def grad_alpha(alpha, delta):
        real_part = jax.grad(lambda a: jnp.real(Y_lm(a, delta, l, m)))(alpha)
        imag_part = jax.grad(lambda a: jnp.imag(Y_lm(a, delta, l, m)))(alpha)
        return real_part + 1j*imag_part

    def grad_delta(alpha, delta):
        real_part = jax.grad(lambda d: jnp.real(Y_lm(alpha, d, l, m)))(delta)
        imag_part = jax.grad(lambda d: jnp.imag(Y_lm(alpha, d, l, m)))(delta)
        return real_part + 1j*imag_part

    return grad_alpha, grad_delta

@partial(jit, static_argnames=('l', 'm'))
def T_lm_scalar(alpha, delta, l, m):

    """
    Computes the toroidal vector spherical harmonic (VSH) T_lm at a single point on the celestial sphere.

    Args:
        alpha (float): Right ascension in radians.
        delta (float): Declination in radians.
        l (int): Degree of the harmonic.
        m (int): Order of the harmonic.

    Returns:
        jnp.ndarray: A 3D complex vector representing the toroidal VSH T_lm at (alpha, delta).

    Notes:
        - Implements the expression: T = (1 / sqrt(l(l+1))).[dY (x) unit radial vector].
        - Uses gradients of the scalar spherical harmonic Y_lm computed with `make_Y_lm_gradients`.
        - Expressed in Cartesian coordinates using the local basis vectors (e_alpha, e_delta).
        - Used for modeling toroidal components of vector fields on the sphere (e.g. rotation-like flows).
        - JIT-compiled and static over (l, m) for efficient reuse.

    Example:
        >>> from src.models.vsh_model import T_lm_scalar
        >>> import jax.numpy as jnp
        >>> result = T_lm_scalar(jnp.pi/7, jnp.pi/3, 10, 2)
        >>> print(f'Real part {result.real}')
        >>> print(f'Imaginary part {result.imag}')
        >>> print(f'Complex vector: {result}')
        >>> # Example of dot product
        >>> dot_result = result.T @ jnp.conj(result)
        >>> print(f'Dot product of result with its complex conjugate {dot_result}')
        >>> # Constructing a Matrix
        >>> v = T_lm_scalar(jnp.pi, jnp.pi/3, 1, 0)
        >>> u = T_lm_scalar(3*jnp.pi, jnp.pi/5, 1, 1)
        >>> M = jnp.vstack([v,u])
        >>> print(f'Constructed matrix M = {M}')
    """

    e_alpha, e_delta = basis_vectors(alpha,delta)

    prefactor = 1/jnp.sqrt(l*(l + 1))

    grad_alpha, grad_delta = make_Y_lm_gradients(l, m)
    Ylm_grad_alpha = grad_alpha(alpha, delta)
    Ylm_grad_delta = grad_delta(alpha, delta)
    safe_cos = jnp.cos(delta)

    return prefactor*(Ylm_grad_delta*e_alpha - (1/safe_cos)*Ylm_grad_alpha*e_delta)


@partial(jit, static_argnames=('l', 'm'))
def T_slm_scalar(alpha, delta, l, m):

    """
    Computes the complex conjugate of the toroidal vector spherical harmonic T_lm at a point.

    Args:
        alpha (float): Right ascension in radians.
        delta (float): Declination in radians.
        l (int): Degree of the harmonic.
        m (int): Order of the harmonic.

    Returns:
        jnp.ndarray: A 3D complex vector representing the conjugate toroidal VSH T_lm at (alpha, delta).

    Notes:
        - Equivalent to `jnp.conj(T_lm_scalar(alpha, delta, l, m))`.
        - Represents the adjoint mode of the toroidal harmonic, useful for inner products and inverse transforms.
        - Shares structure and normalisation with `T_lm_scalar`, but with reversed complex phase.

    Example:
        >>> from src.models.vsh_model import T_slm_scalar
        >>> import jax.numpy as jnp
        >>> result = T_slm_scalar(jnp.pi/7, jnp.pi/3, 10, 2)
        >>> print(f'Real part {result.real}')
        >>> print(f'Imaginary part {result.imag}')
        >>> print(f'Complex vector: {result}')
        >>> # Example of dot product
        >>> dot_result = result.T @ jnp.conj(result)
        >>> print(f'Dot product of result with its complex conjugate {dot_result}')
        >>> # Constructing a Matrix
        >>> v = T_slm_scalar(jnp.pi, jnp.pi/3, 1, 0)
        >>> u = T_slm_scalar(3*jnp.pi, jnp.pi/5, 1, 1)
        >>> M = jnp.vstack([v,u])
        >>> print(f'Constructed matrix M = {M}')

    """

    return jnp.conj(T_lm_scalar(alpha, delta, l, m))

@partial(jit, static_argnames=('l', 'm', 'grid'))
def T_lm(alpha, delta, l, m, grid=True):

    """
    Computes the toroidal vector spherical harmonic T_lm over scalar inputs or a coordinate grid.

    Args:
        alpha (float or jnp.ndarray): Right ascension(s) in radians. Can be scalar, 1D, or 2D array.
        delta (float or jnp.ndarray): Declination(s) in radians. Same shape rules as `alpha`.
        l (int): Degree of the harmonic.
        m (int): Order of the harmonic.
        grid (bool, optional): 
            - If True (default), treats `alpha` and `delta` as 1D arrays and evaluates over a meshgrid.
            - If False, assumes paired (alpha_i, delta_i) values and evaluates pointwise.

    Returns:
        jnp.ndarray: Complex-valued 3D vector field representing T_lm evaluated at the given coordinates.

    Notes:
        - Wraps `T_lm_scalar()` with `jax.vmap` or `jnp.meshgrid` for efficient batch evaluation.
        - JIT-compiled and static over (l, m, grid) for performance.
        - Used in constructing full VSH expansions involving toroidal components.

    Example:
        >>> from src.models.vsh_model import T_lm
        >>> import jax.numpy as jnp
        >>> alpha0, delta0 = jnp.array([0.2, 1.4, -6.4]), jnp.array([-5.236, 8.75, 5.8])
        >>> alpha1, delta1 = jnp.array([-1.335, 9.21, 8.2]), jnp.array([-5.2, -7.42, 0.2356])
        >>> v = T_lm(alpha0, delta0, 4, 0, grid = False)
        >>> u = T_lm(alpha1, delta1, 6, 2, grid = False)
        >>> v_u = v @ u
        >>> u_v = u @ v
        >>> print(f'vT.u = {v_u}')
        >>> print(f'uT.v = {u_v}')
    """

    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return T_lm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: T_lm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: T_lm_scalar(a, d, l, m))(alpha, delta)



@partial(jit, static_argnames=('l', 'm', 'grid'))
def T_slm(alpha, delta, l, m, grid=True):

    """
    Computes the complex conjugate of the toroidal vector spherical harmonic T_lm over input coordinates.

    Args:
        alpha (float or jnp.ndarray): Right ascension(s) in radians. Can be scalar, 1D, or 2D array.
        delta (float or jnp.ndarray): Declination(s) in radians. Same shape rules as `alpha`.
        l (int): Degree of the harmonic.
        m (int): Order of the harmonic.
        grid (bool, optional): 
            - If True (default), treats `alpha` and `delta` as 1D arrays and evaluates over a meshgrid.
            - If False, assumes matched arrays of (alpha, delta) points.

    Returns:
        jnp.ndarray: Complex-valued 3D vector field representing the conjugate T_lm at the given points.

    Notes:
        - Computes the conjugate of `T_lm_scalar`, vectorized with `jax.vmap` as needed.
        - Matches the structure and grid behavior of `T_lm`.
        - Useful for forming Hermitian inner products in VSH analyses.

    Example:
        >>> from src.models.vsh_model import T_slm
        >>> import jax.numpy as jnp
        >>> alpha0, delta0 = jnp.array([0.2, 1.4, -6.4]), jnp.array([-5.236, 8.75, 5.8])
        >>> alpha1, delta1 = jnp.array([-1.335, 9.21, 8.2]), jnp.array([-5.2, -7.42, 0.2356])
        >>> v = T_slm(alpha0, delta0, 4, 0, grid = False)
        >>> u = T_slm(alpha1, delta1, 6, 2, grid = False)
        >>> v_u = v @ u
        >>> u_v = u @ v
        >>> print(f'vT.u = {v_u}')
        >>> print(f'uT.v = {u_v}')
    """

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
    Computes the spheroidal vector spherical harmonic (VSH) S_lm at a single point on the celestial sphere.

    Args:
        alpha (float): Right ascension in radians.
        delta (float): Declination in radians.
        l (int): Degree of the harmonic.
        m (int): Order of the harmonic.

    Returns:
        jnp.ndarray: A 3D complex vector representing the spheroidal VSH S_lm at (alpha, delta).

    Notes:
        - Computes the tangential vector field S = (1 / sqrt(l(l+1))) · dY_lm, projected onto the spherical basis.
        - Uses `make_Y_lm_gradients` to obtain partial derivatives of the spherical harmonic Y_lm.
        - Converts gradients into a 3D Cartesian vector using `basis_vectors` e_alpha and e_delta.
        - Models "glide-like" vector fields (e.g., those resulting from dipole acceleration patterns).
        - JIT-compiled and static over (l, m) for performance.
    
    Example:
        >>> from src.models.vsh_model import S_lm_scalar
        >>> import jax.numpy as jnp
        >>> result = S_lm_scalar(jnp.pi/7, jnp.pi/3, 10, 2)
        >>> print(f'Real part {result.real}')
        >>> print(f'Imaginary part {result.imag}')
        >>> print(f'Complex vector: {result}')
        >>> # Example of dot product
        >>> dot_result = result.T @ jnp.conj(result)
        >>> print(f'Dot product of result with its complex conjugate {dot_result}')
        >>> # Constructing a Matrix
        >>> v = S_lm_scalar(jnp.pi, jnp.pi/3, 1, 0)
        >>> u = S_lm_scalar(3*jnp.pi, jnp.pi/5, 1, 1)
        >>> M = jnp.vstack([v,u])
        >>> print(f'Constructed matrix M = {M}')

    """

    e_alpha, e_delta = basis_vectors(alpha,delta)

    prefactor = 1/jnp.sqrt(l*(l + 1))

    grad_alpha, grad_delta = make_Y_lm_gradients(l, m)
    Ylm_grad_alpha = grad_alpha(alpha, delta)
    Ylm_grad_delta = grad_delta(alpha, delta)
    safe_cos = jnp.cos(delta)

    return prefactor*((1/safe_cos)*Ylm_grad_alpha*e_alpha + Ylm_grad_delta*e_delta)


@partial(jit, static_argnames=('l', 'm'))
def S_slm_scalar(alpha, delta, l, m):

    """
    Computes the complex conjugate of the spheroidal vector spherical harmonic S_lm at a single sky location.

    Args:
        alpha (float): Right ascension in radians.
        delta (float): Declination in radians.
        l (int): Degree of the harmonic.
        m (int): Order of the harmonic.

    Returns:
        jnp.ndarray: A 3D complex vector giving the conjugate of S_lm evaluated at (alpha, delta).

    Notes:
        - Defined as the complex conjugate of `S_lm_scalar(alpha, delta, l, m)`.
        - Shares structure and normalization with the toroidal mode counterparts.
        - Useful for computing adjoint fields or Hermitian products in VSH estimation and fitting.

    Example:
        >>> from src.models.vsh_model import S_slm_scalar
        >>> import jax.numpy as jnp
        >>> result = S_slm_scalar(jnp.pi/7, jnp.pi/3, 10, 2)
        >>> print(f'Real part {result.real}')
        >>> print(f'Imaginary part {result.imag}')
        >>> print(f'Complex vector: {result}')
        >>> # Example of dot product
        >>> dot_result = result.T @ jnp.conj(result)
        >>> print(f'Dot product of result with its complex conjugate {dot_result}')
        >>> # Constructing a Matrix
        >>> v = S_slm_scalar(jnp.pi, jnp.pi/3, 1, 0)
        >>> u = S_slm_scalar(3*jnp.pi, jnp.pi/5, 1, 1)
        >>> M = jnp.vstack([v,u])
        >>> print(f'Constructed matrix M = {M}')

    """

    return jnp.conj(S_lm_scalar(alpha, delta, l, m))

@partial(jit, static_argnames=('l', 'm', 'grid'))
def S_lm(alpha, delta, l, m, grid=True):

    """
    Evaluates the spheroidal vector spherical harmonic S_lm over a scalar point or coordinate grid.

    Args:
        alpha (float or jnp.ndarray): Right ascension(s) in radians. Can be scalar, 1D, or 2D array.
        delta (float or jnp.ndarray): Declination(s) in radians. Same shape rules as `alpha`.
        l (int): Degree of the harmonic.
        m (int): Order of the harmonic.
        grid (bool, optional): 
            - If True (default), treats `alpha` and `delta` as 1D arrays and evaluates on a meshgrid.
            - If False, assumes element-wise paired arrays (alpha_i, delta_i).

    Returns:
        jnp.ndarray: Complex-valued 3D vector field representing the spheroidal harmonic S_lm evaluated at the input coordinates.

    Notes:
        - Wraps `S_lm_scalar()` and applies `jax.vmap` or `jnp.meshgrid` to support batch evaluation.
        - Used to represent "glide" components in vector field decompositions (e.g., from acceleration).
        - JIT-compiled with static (l, m, grid) for high-performance evaluation.

    Example:
        >>> from src.models.vsh_model import S_lm
        >>> import jax.numpy as jnp
        >>> alpha0, delta0 = jnp.array([0.2, 1.4, -6.4]), jnp.array([-5.236, 8.75, 5.8])
        >>> alpha1, delta1 = jnp.array([-1.335, 9.21, 8.2]), jnp.array([-5.2, -7.42, 0.2356])
        >>> v = S_lm(alpha0, delta0, 4, 0, grid = False)
        >>> u = S_lm(alpha1, delta1, 6, 2, grid = False)
        >>> v_u = v @ u
        >>> u_v = u @ v
        >>> print(f'vT.u = {v_u}')
        >>> print(f'uT.v = {u_v}')
    """

    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return S_lm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: S_lm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: S_lm_scalar(a, d, l, m))(alpha, delta)



@partial(jit, static_argnames=('l', 'm', 'grid'))
def S_slm(alpha, delta, l, m, grid=True):

    """
    Evaluates the complex conjugate of the spheroidal vector spherical harmonic S_lm over a point or coordinate grid.

    Args:
        alpha (float or jnp.ndarray): Right ascension(s) in radians. Can be scalar, 1D, or 2D array.
        delta (float or jnp.ndarray): Declination(s) in radians. Same shape rules as `alpha`.
        l (int): Degree of the harmonic.
        m (int): Order of the harmonic.
        grid (bool, optional): 
            - If True (default), treats `alpha` and `delta` as 1D arrays and evaluates on a meshgrid.
            - If False, evaluates over element-wise paired (alpha, delta) values.

    Returns:
        jnp.ndarray: Complex-valued 3D vector field representing the conjugate of S_lm at the input coordinates.

    Notes:
        - Computes `jnp.conj(S_lm_scalar(...))` in a batched way.
        - Mirrors the interface and structure of `S_lm`.
        - Useful in constructing inner products and adjoint operators for VSH fitting.
    
    Example:
        >>> from src.models.vsh_model import S_slm
        >>> import jax.numpy as jnp
        >>> alpha0, delta0 = jnp.array([0.2, 1.4, -6.4]), jnp.array([-5.236, 8.75, 5.8])
        >>> alpha1, delta1 = jnp.array([-1.335, 9.21, 8.2]), jnp.array([-5.2, -7.42, 0.2356])
        >>> v = S_slm(alpha0, delta0, 4, 0, grid = False)
        >>> u = S_slm(alpha1, delta1, 6, 2, grid = False)
        >>> v_u = v @ u
        >>> u_v = u @ v
        >>> print(f'vT.u = {v_u}')
        >>> print(f'uT.v = {u_v}')
    """

    if jnp.ndim(alpha) == 0 and jnp.ndim(delta) == 0:
        return S_slm_scalar(alpha, delta, l, m)

    if grid:
        A, D = jnp.meshgrid(alpha, delta, indexing='ij')
        return jax.vmap(jax.vmap(lambda a, d: S_slm_scalar(a, d, l, m)))(A, D)
    else:
        return jax.vmap(lambda a, d: S_slm_scalar(a, d, l, m))(alpha, delta)
    
@partial(jit, static_argnames=('grid'))
def toy_model_l_1(alpha, delta, theta, grid):
    """
    Computes a toy vector spherical harmonic (VSH) model of degree l = 1 using toroidal and spheroidal components.

    Args:
        alpha (float or jnp.ndarray): Right ascension(s) in radians.
        delta (float or jnp.ndarray): Declination(s) in radians.
        theta (array-like): Length-6 parameter array containing:
            [t_10, t_11_real, t_11_imag, s_10, s_11_real, s_11_imag].
        grid (bool): 
            - If True, evaluates over a 2D meshgrid formed from `alpha` and `delta`.
            - If False, evaluates at paired (alpha_i, delta_i) positions.

    Returns:
        jnp.ndarray: A 3D complex vector field computed from the l = 1 toroidal and spheroidal VSH components.

    Notes:
        - Represents a simplified model used to describe low-degree vector fields on the celestial sphere.
        - Used as a test model for fitting proper motion fields such as those from Gaia QSO observations.
        - Each parameter in `theta` scales a VSH mode: real and imaginary parts of m = 1 terms are handled explicitly.
    """

    a = alpha
    d = delta
    l=1
    V = (theta[0]*T_lm(a, d, l, 0, grid=grid) + 
         theta[1]*jnp.real(T_lm(a,d,l,1,grid=grid)) -
         theta[2]*jnp.imag(T_lm(a,d,l,1,grid=grid)) + 
         theta[3]*S_lm(a,d,l,0,grid=grid) + 
         theta[4]*jnp.real(S_lm(a,d,l,1,grid=grid)) - 
         theta[5]*jnp.imag(S_lm(a,d,l,1,grid=grid)) 
         ) 
    return V

@jax.jit
def toy_least_square(angles, obs, error, t_10, t_11r, t_11i, s_10, s_11r, s_11i):
    """
    Evaluates a least squares loss function for fitting an l = 1 vector spherical harmonic model to observed proper motions.

    Args:
        angles (Tuple[jnp.ndarray, jnp.ndarray]):
            - `alpha`: Array of right ascensions in radians.
            - `delta`: Array of declinations in radians.
        obs (Tuple[jnp.ndarray, jnp.ndarray]):
            - `mu_alpha_obs`: Observed proper motions in RA (mu_alpha*).
            - `mu_delta_obs`: Observed proper motions in Dec (mu_delta).
        error (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            - `sigma_mu_alpha`: Proper motion error in RA.
            - `sigma_mu_delta`: Proper motion error in Dec.
            - `rho`: Correlation coefficient between mu_alpha* and mu_delta.
        t_10, t_11r, t_11i, s_10, s_11r, s_11i (float): 
            Parameters of the VSH model for l = 1 (6 total: toroidal and spheroidal real and imaginary parts).

    Returns:
        float: Total least-squares loss value across all sources.

    Notes:
        - Uses the `toy_model_l_1` VSH model to compute predicted proper motions.
        - Computes residuals and propagates proper motion uncertainties using full covariance (A matrix).
        - Suitable for optimisation routines (e.g., Minuit) to estimate best-fit l = 1 VSH coefficients.
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
    Computes the full vector spherical harmonic (VSH) model up to degree `lmax` using given coefficients.

    Args:
        alpha (float or jnp.ndarray): Right ascension(s) in radians.
        delta (float or jnp.ndarray): Declination(s) in radians.
        theta (array-like): Flattened array of VSH coefficients. For each (l, m), includes:
            - t_lm and s_lm for m = 0 (real scalars),
            - Re(t_lm), Im(t_lm), Re(s_lm), Im(s_lm) for m > 0.
            Total length = 2 * lmax * (lmax + 2).
        lmax (int): Maximum spherical harmonic degree to include in the model.
        grid (bool, optional): 
            - If True, evaluates over a meshgrid formed from `alpha` and `delta`.
            - If False, assumes paired (alpha_i, delta_i) input arrays.

    Returns:
        jnp.ndarray: Complex 3D vector field evaluated at each coordinate using VSH basis functions.

    Notes:
        - Accumulates toroidal and spheroidal components across all degrees 1 to lmax.
        - Modeled field is constructed from linear combinations of T_lm and S_lm components.
        - Coefficients are indexed according to VSH convention and mapped by (l, m).
    """

    a = alpha
    d = delta
    V = jnp.zeros(3) 

    index = 0
    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            T = T_lm(a, d, l, m, grid=grid)
            S = S_lm(a, d, l, m, grid=grid)

            if m == 0:
                t_lm = theta[index]
                s_lm = theta[index + 1]
                index += 2
                V += t_lm * jnp.real(T) # remove real if wrong
                V += s_lm * jnp.real(S)
            else:
                t_r, t_i = theta[index], theta[index + 1]
                s_r, s_i = theta[index + 2], theta[index + 3]
                index += 4

                V += t_r*jnp.real(T) - t_i*jnp.imag(T)
                V += s_r*jnp.real(S) - s_i*jnp.imag(S)

    return V

def count_vsh_coeffs(lmax):

    """
    Computes the number of parameters given lmax
    Args:
        lmax (int), largest degree selsected

    Returns:
        int: number of free parameters (t_lm + s_lm)

    Example:
        >>> from src.models.vsh_model import count_vsh_coeffs
        >>> n_param = count_vsh_coeffs(2)
        >>> print(n_param)
    """

    return 2*lmax*(lmax + 2)

@partial(jit, static_argnames=['lmax', 'grid'])
def least_square(angles, obs, error, theta, lmax, grid):

    """
    Computes the total least squares loss between observed proper motions and a VSH model prediction up to `lmax`.

    Args:
        angles (Tuple[jnp.ndarray, jnp.ndarray]):
            - `alpha`: Array of right ascensions in radians.
            - `delta`: Array of declinations in radians.
        obs (Tuple[jnp.ndarray, jnp.ndarray]):
            - `mu_alpha_obs`: Observed proper motions in RA (mu_alpha*).
            - `mu_delta_obs`: Observed proper motions in Dec (mu_delta).
        error (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            - `sigma_mu_alpha`: Uncertainty in RA proper motion.
            - `sigma_mu_delta`: Uncertainty in Dec proper motion.
            - `rho`: Correlation between RA and Dec components.
        theta (jnp.ndarray): Flattened array of VSH coefficients used to compute the modeled field.
        lmax (int): Maximum spherical harmonic degree used in the model.
        grid (bool, optional): 
            - If True, evaluates over 2D meshgrid of coordinates.
            - If False, evaluates paired positions element-wise.

    Returns:
        float: Sum of weighted squared residuals over all sources.

    Notes:
        - Projects the modeled vector field onto the local (e_alpha, e_delta) basis at each point.
        - Uses full covariance matrix A for error propagation and residual weighting.
        - Intended for use in fitting the VSH model to astrometric proper motion data (e.g., from Gaia).
    """

    alpha, delta = angles
    mu_a_obs, mu_d_obs = obs
    s_mu_a, s_mu_d, rho = error

    def per_point(alpha_i, delta_i, mu_a_i, mu_d_i, s_a, s_d, r):
        e_a, e_d = basis_vectors(alpha_i, delta_i)

        V = model_vsh(alpha_i, delta_i, theta, lmax=lmax, grid=grid)
        V_alpha = jnp.vdot(V, e_a)
        V_delta = jnp.vdot(V, e_d)

        d_alpha = mu_a_i - V_alpha
        d_delta = mu_d_i - V_delta

        norm_alpha = d_alpha / s_a
        norm_delta = d_delta / s_d

        X2 = (norm_alpha**2 - 2*r*norm_alpha*norm_delta + norm_delta**2) / (1 - r**2)
        return X2

    batched_fn = vmap(per_point)
    losses = batched_fn(alpha, delta, mu_a_obs, mu_d_obs, s_mu_a, s_mu_d, rho)
    return jnp.sum(losses)

@partial(jit, static_argnames=['lmax', 'grid'])
def model_vsh_hmc(alpha, delta, theta_t, theta_s, lmax, grid):
    """
    Computes the full vector spherical harmonic (VSH) model up to degree `lmax` using given coefficients.

    Args:
        alpha (float or jnp.ndarray): Right ascension(s) in radians.
        delta (float or jnp.ndarray): Declination(s) in radians.
        theta (array-like): Flattened array of VSH coefficients. For each (l, m), includes:
            - t_lm and s_lm for m = 0 (real scalars),
            - Re(t_lm), Im(t_lm), Re(s_lm), Im(s_lm) for m > 0.
            Total length = 2 * lmax * (lmax + 2).
        lmax (int): Maximum spherical harmonic degree to include in the model.
        grid (bool, optional): 
            - If True, evaluates over a meshgrid formed from `alpha` and `delta`.
            - If False, assumes paired (alpha_i, delta_i) input arrays.

    Returns:
        jnp.ndarray: Complex 3D vector field evaluated at each coordinate using VSH basis functions.

    Notes:
        - Accumulates toroidal and spheroidal components across all degrees 1 to lmax.
        - Modeled field is constructed from linear combinations of T_lm and S_lm components.
        - Coefficients are indexed according to VSH convention and mapped by (l, m).
    """

    a = alpha
    d = delta
    V = jnp.zeros(3, dtype=jnp.complex64)

    index_t = 0
    index_s = 0
    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            T = T_lm(a, d, l, m, grid=grid)
            S = S_lm(a, d, l, m, grid=grid)

            if m == 0:
                t_lm = theta_t[index_t]
                s_lm = theta_s[index_s]
                index_t += 1
                index_s += 1
                V += t_lm * T
                V += s_lm * S
            else:
                t_r, t_i = theta_t[index_t], theta_t[index_t + 1]
                s_r, s_i = theta_s[index_s], theta_s[index_s + 1]
                index_t += 2
                index_s += 2

                V += t_r*jnp.real(T) - t_i*jnp.imag(T)
                V += s_r*jnp.real(S) - s_i*jnp.imag(S)

    return V

@partial(jit, static_argnames=['lmax', 'grid'])
def least_square_hmc(angles, obs, error, theta_t, theta_s, lmax, grid):

    """
    Computes the total least squares loss between observed proper motions and a VSH model prediction up to `lmax`.

    Args:
        angles (Tuple[jnp.ndarray, jnp.ndarray]):
            - `alpha`: Array of right ascensions in radians.
            - `delta`: Array of declinations in radians.
        obs (Tuple[jnp.ndarray, jnp.ndarray]):
            - `mu_alpha_obs`: Observed proper motions in RA (mu_alpha*).
            - `mu_delta_obs`: Observed proper motions in Dec (mu_delta).
        error (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            - `sigma_mu_alpha`: Uncertainty in RA proper motion.
            - `sigma_mu_delta`: Uncertainty in Dec proper motion.
            - `rho`: Correlation between RA and Dec components.
        theta (jnp.ndarray): Flattened array of VSH coefficients used to compute the modeled field.
        lmax (int): Maximum spherical harmonic degree used in the model.
        grid (bool, optional): 
            - If True, evaluates over 2D meshgrid of coordinates.
            - If False, evaluates paired positions element-wise.

    Returns:
        float: Sum of weighted squared residuals over all sources.

    Notes:
        - Projects the modeled vector field onto the local (e_alpha, e_delta) basis at each point.
        - Uses full covariance matrix A for error propagation and residual weighting.
        - Intended for use in fitting the VSH model to astrometric proper motion data (e.g., from Gaia).
    """

    alpha, delta = angles
    mu_a_obs, mu_d_obs = obs
    s_mu_a, s_mu_d, rho = error

    def per_point(alpha_i, delta_i, mu_a_i, mu_d_i, s_a, s_d, r):
        e_a, e_d = basis_vectors(alpha_i, delta_i)

        A = jnp.array([
            [s_a**2, r*s_a*s_d],
            [r*s_a*s_d, s_d**2]
        ])

        V = model_vsh_hmc(alpha_i, delta_i, theta_t, theta_s, lmax=lmax, grid=grid)
        V_alpha = jnp.vdot(V, e_a).real
        V_delta = jnp.vdot(V, e_d).real

        D = jnp.array([mu_a_i - V_alpha, mu_d_i - V_delta])
        x = jnp.linalg.solve(A, D)
        return D @ x

    batched_fn = vmap(per_point)
    losses = batched_fn(alpha, delta, mu_a_obs, mu_d_obs, s_mu_a, s_mu_d, rho)
    return jnp.sum(losses)

@partial(jit, static_argnames = ['lmax'])
def compute_X2(alpha, delta, mu_a_obs, mu_d_obs, s_mu_a, s_mu_d, rho, theta, lmax):
    """Compute X^2 residuals for each source."""

    def per_point(alpha_i, delta_i, mu_a_i, mu_d_i, s_a, s_d, r):
        e_a, e_d = basis_vectors(alpha_i, delta_i)
        A = jnp.array([
            [s_a**2, r * s_a * s_d],
            [r * s_a * s_d, s_d**2]
        ])
        V = model_vsh(alpha_i, delta_i, theta, lmax, grid=False)
        V_alpha = jnp.vdot(V, e_a).real
        V_delta = jnp.vdot(V, e_d).real

        D = jnp.array([mu_a_i - V_alpha, mu_d_i - V_delta])
        return D.T @ jnp.linalg.inv(A) @ D

    batched_fn = jnp.vectorize(per_point, signature='(),(),(),(),(),(),()->()')
    return batched_fn(alpha, delta, mu_a_obs, mu_d_obs, s_mu_a, s_mu_d, rho)
