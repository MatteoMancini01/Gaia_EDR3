from iminuit import Minuit
from src.models.vsh_model import*
import numpy as np
import jax.numpy as jnp
import gc

def least_square_clip(angles, obs, error, theta_init, lmax = 3, kappa=3.0, max_iter=10):

    """
    Performs robust least-squares fitting of vector spherical harmonics (VSH) to proper motion data 
    with iterative outlier rejection.

    This function applies an iterative clipping procedure to fit a VSH model to observed proper motions, 
    removing outliers based on a threshold on normalized residuals (X^2). The model parameters are 
    optimized using the Minuit minimizer, and convergence is determined by the stability of the set of 
    outlier-rejected sources across iterations.

    Args:
        angles (Tuple[jnp.ndarray, jnp.ndarray]): Tuple of (alpha, delta) in radians — the sky coordinates 
            of the sources.
        obs (Tuple[jnp.ndarray, jnp.ndarray]): Tuple of (mu_alpha_obs, mu_delta_obs) — the observed proper 
            motions in mas/yr.
        error (Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]): Tuple of (sigma_mu_alpha, sigma_mu_delta, rho), 
            representing uncertainties and correlation in the proper motions.
        theta_init (jnp.ndarray): Initial guess for the VSH parameters (flattened array).
        lmax (int, optional): Maximum degree of the VSH expansion. Default is 3.
        kappa (float, optional): Clipping threshold multiplier for residuals. Sources with normalized 
            residuals > kappa * median are excluded. Default is 3.0.
        max_iter (int, optional): Maximum number of clipping iterations. Default is 10.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - Final estimated VSH parameter vector (theta).
            - Boolean mask (keep) indicating which sources were retained after clipping.

    Notes:
        - VSH parameters are interpreted as complex coefficients and converted to Cartesian frame using 
          normalization constants.
        - Uses Minuit to perform least squares optimization with errordef = Minuit.LEAST_SQUARES.
        - `jax.clear_caches()` is called at each iteration to manage JAX memory.
        - Outlier rejection is based on chi-squared residuals of all sources (not just the current inliers).
    """

    
    alpha, delta = angles
    mu_a_obs, mu_d_obs = obs
    s_mu_a, s_mu_d, rho = error

    keep = jnp.ones_like(alpha, dtype=bool)
    theta = theta_init

    prev_outliers = None

    for iteration in range(max_iter):
        print('Iteration:', iteration+1)
        alpha_k, delta_k = alpha[keep], delta[keep]
        obs_k = (mu_a_obs[keep], mu_d_obs[keep])
        err_k = (s_mu_a[keep], s_mu_d[keep], rho[keep])
        angles_k = (alpha_k, delta_k)

        def least_square_wrapper(*theta_flat):
            theta_arr = jnp.array(theta_flat)
            return least_square(angles_k, obs_k, err_k, theta_arr, lmax=lmax, grid=False)

        m = Minuit(least_square_wrapper, *theta)
        m.errordef = Minuit.LEAST_SQUARES

        m.migrad()

        theta = jnp.array([m.values[name] for name in m.parameters])

        C0 = 1000/np.sqrt(8*np.pi/3)
        C1 = 1000/np.sqrt(4*np.pi/3)

        print(f'Current g components [μas/yr]: gx = {-theta[4]*C1}, gy = {theta[5]*C1}, gz = {theta[1]*C0}')

        del m
        gc.collect()
        jax.clear_caches()

        # Compute X^2 over full dataset (not just kept subset)
        X = np.sqrt(compute_X2(alpha, delta, mu_a_obs, mu_d_obs, s_mu_a, s_mu_d, rho, theta, lmax))
        median_X = jnp.median(X)
        keep = X < kappa*median_X

        print(f"Rejected: {(~keep).sum()} sources")

        if prev_outliers is not None and jnp.array_equal(keep, prev_outliers):
            print(f"Converged after {iteration+1} iterations.")
            break
        prev_outliers = keep
        print(f'Length of keep array: {len(keep)}')

    return theta, keep
