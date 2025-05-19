import jax
import pandas as pd
import jax.numpy as jnp
from models.vsh_model import basis_vectors, count_vsh_coeffs, model_vsh
from jax import random
import numpy as np

def generate_random_theta(lmax, amplitude=0.01, seed=0):
    key = jax.random.PRNGKey(seed)
    n_params = count_vsh_coeffs(lmax)
    theta = jax.random.uniform(key, shape=(n_params,), minval=-amplitude, maxval=amplitude)
    return theta

lmax = 2
# Choose fixed t_lm and s_lm values (mas/yr)
theta_gen = generate_random_theta(lmax, amplitude=0.06, seed=0)

key = random.PRNGKey(0)

# Generate N random points on the sphere (RA, Dec in radians)
N = 100000

ra = random.uniform(key, shape=(N,), minval=0.0, maxval=2 * jnp.pi)
dec = jnp.arcsin(random.uniform(key, shape=(N,), minval=-1.0, maxval=1.0))  # uniform on sphere

angles_gen = jnp.stack([ra, dec])  # shape (2, N)

# Use model to get proper motion vectors, then project to RA/Dec components
mu_alpha = []
mu_delta = []

for i in range(N):
    alpha_i = ra[i]
    delta_i = dec[i]
    e_a, e_d = basis_vectors(alpha_i, delta_i)

    V = model_vsh(alpha_i, delta_i, theta_gen, lmax, grid=False)
    mu_alpha.append(jnp.vdot(V, e_a).real)
    mu_delta.append(jnp.vdot(V, e_d).real)

mu_alpha = jnp.array(mu_alpha)  # shape (N,)
mu_delta = jnp.array(mu_delta)

# Add Gaussian noise
noise_level = 0.03  # mas/yr
key1, key2 = random.split(key)
mu_alpha_noisy = mu_alpha + random.normal(key1, shape=(N,)) * noise_level
mu_delta_noisy = mu_delta + random.normal(key2, shape=(N,)) * noise_level

# Pack into obs and error arrays
obs = jnp.stack([mu_alpha_noisy, mu_delta_noisy])  # shape (2, N)
error = jnp.stack([
    jnp.ones(N) * noise_level,       # pmra_error
    jnp.ones(N) * noise_level,       # pmdec_error
    jnp.zeros(N)                    # pmra_pmdec_corr
])

# Convert angles back to degrees for clarity (optional)
ra_deg = jnp.degrees(ra)
dec_deg = jnp.degrees(dec)

# Unpack observations and errors
pmra = obs[0]
pmdec = obs[1]
pmra_error = error[0]
pmdec_error = error[1]
pmra_pmdec_corr = error[2]

# Create dictionary for DataFrame
data = {
    'ra': ra_deg,
    'dec': dec_deg,
    'pmra': pmra,
    'pmdec': pmdec,
    'pmra_error': pmra_error,
    'pmdec_error': pmdec_error,
    'pmra_pmdec_corr': pmra_pmdec_corr
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('../../fake_data/fake_vsh_data.csv', index=False)

np.save("../../fake_data/fake_data/theta_true.npy", np.array(theta_gen))
