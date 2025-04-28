import jax.numpy as jnp
import jax 

class VSHModel:

    """
    A class designed for Vector Spherical Harmonics (VSH)
    VSHModel performs a VSH expansion of a proper motion field on the celestial sphere.
    It models the observed proper motions of extragalactic sources as a sum of toroidal and spheroidal (rotation and glide respectively) 
    components up to a maximum spherical harmonic degree l_max.

    The objective is to estimate the glide terms, which are directly related to the acceleration of 
    the Solar system barycentre relative to the distant Universe.

    Main features:
    -
    -

    """

    def __init__(self, alpha, delta, mu_alpha_star,
                 mu_delta, sigma_mu_alpha_star, sigma_mu_delta,
                 rho=None, l_max=3, clip_limit = 3.0):
        
        """
        Initialising input

        Paraneters:
        ----------
        alpha : jnp.array of source right ascensions (radians)
        delta : jnp.array of source declinations (radians)
        mu_alpha_star : jnp.array of proper motions in RA ($\mu\_{\alpha*}$)
        mu_delta : jnp.array of proper motions in Dec ($\mu\_{\delta}$)
        sigma_mu_alpha_star : jnp.array of errors on proper motion in RA ($\sigma_{\mu\_{\alpha*}}$)
        sigma_mu_delta : jnp.array of errors on proper motion in Dec ($\sigma_{\mu\_{\delta}}$)
        rho : Correlation coefficient between proper motions in RA and Dec for each source (default `None`)
        l_max : Maximum VSH degree $l_{max}$ we want to fit (default = 3)
        clip_limit : Outlier rejection threshold (default = 3.0)

        """
        
        self.alpha = alpha
        self.delta = delta
        self.mu_delta = mu_delta
        self.mu_alpha_star = mu_alpha_star
        self.sigma_mu_alpha_star = sigma_mu_alpha_star
        self.sigma_mu_delta = sigma_mu_delta
        self.rho = rho
        self.l_max = l_max
        self.clip_limit = clip_limit
