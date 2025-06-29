import corner
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
from astropy import units as u
import arviz as az

def traceplot_check(mcmc):

    """
    Generates trace plots for all sampled parameters and a focused trace plot
    for the spheroidal dipole components (s_10, s_11r, s_11i) using ArviZ.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        A fitted NumPyro MCMC object containing posterior samples.

    Notes
    -----
    This function visualizes:
    - Trace plots for all parameters sampled in the MCMC run.
    - A focused trace plot for theta indices [1, 4, 5], corresponding to
      the spheroidal VSH dipole components under standard indexing.

    Example
    -------
    >>> from numpyro.infer import MCMC, NUTS
    >>> kernel = NUTS(my_model)
    >>> mcmc = MCMC(kernel, num_samples=5000, num_warmup=1000)
    >>> mcmc.run(rng_key, ...)
    >>> traceplot_check(mcmc)
    """


    idata = az.from_numpyro(mcmc)
    az.plot_trace(idata)
    az.plot_trace(idata, coords={"theta_dim_0": [1, 4, 5]})
    
    plt.tight_layout()
    plt.show()

def corner_plot_eq_comp(posterior_sample, save = None, bins = 100, smooth = 1):

    """
    Creates a corner plot of the equatorial Cartesian components of the glide vector (g_z, g_x, g_y).

    Parameters
    ----------
    posterior_sample : dict
        Dictionary containing posterior samples (with 'theta' key) from HMC or MCMC.
    save : str or None, optional
        Filename to save the plot (without extension). If None, the plot is not saved.
    bins : int, optional
        Number of bins for histograms (default is 100).
    smooth : float, optional
        Gaussian smoothing for the histograms (default is 1).

    Example
    -------
    >>> corner_plot_eq_comp(posterior_sample, save="glide_components")
    """

    ps = posterior_sample

    # Define C0 and C1 for convertion to equatorial coordinates

    C0 = np.sqrt(8*np.pi/3)/1000
    C1 = np.sqrt(4*np.pi/3)/1000

    samples_array = np.column_stack([ps[:,1]/C0, -ps[:,4]/C1, ps[:,5]/C1])

    fig = plt.figure(figsize=(6, 6))  
    corner.corner(samples_array, 
                    labels = [r"$g_z$", r"$g_x$", r"$g_y$"],
                    quantiles=[0.16, 0.5, 0.84], 
                    show_titles=True,
                    title_kwargs={"fontsize":14},
                    label_kwargs={"fontsize":14},
                    bins=bins,
                    color="navy",
                    smooth=smooth,
                    fig=fig)
    plt.tight_layout()

    if save:
        plt.savefig(f"plots/main_plots/{save}.png", dpi=300, bbox_inches='tight')

    plt.show()

def corner_plot_ra_dec(posterior_sample, smooth = 1, save = None, bins = 100):

    """
    Creates a corner plot showing the RA and Dec (in degrees) of the inferred glide vector direction.

    Parameters
    ----------
    posterior_sample : dict
        Posterior samples containing 'theta' key.
    smooth : float, optional
        Gaussian smoothing for the histograms.
    save : str or None, optional
        Filename (without extension) to save the plot. Default is None (no saving).
    bins : int, optional
        Number of bins for histograms (default is 100).

    Example
    -------
    >>> corner_plot_ra_dec(posterior_sample, smooth=1, save="glide_direction")
    """

    ps = posterior_sample

    # Define C0 and C1 for convertion to equatorial coordinates
    C0 = np.sqrt(8*np.pi/3)/1000
    C1 = np.sqrt(4*np.pi/3)/1000

    gz = ps[:,1]/C0
    gx = -ps[:,4]/C1
    gy = ps[:,5]/C1

    g_mag2 = gx**2 + gy**2 + gz**2

    # Covert vec g components into ra, dec

    ra_rad = np.arctan2(gy,gx) % (2*np.pi)
    #dec_rad = np.arcsin(np.clip(gz / np.sqrt(g_mag2), -1.0, 1.0))
    dec_rad = np.arcsin(gz/np.sqrt(g_mag2))

    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.rad2deg(dec_rad)

    sample_arr = np.column_stack([ra_deg, dec_deg])

    fig = plt.figure(figsize=(6, 6))
    corner.corner(sample_arr,
                  labels = ['Ra (deg)', 'Dec (deg)'],
                  quantiles=[0.16, 0.5, 0.84], # 68% credible interval
                  show_titles=True,
                  title_kwargs={"fontsize":10},
                  label_kwargs={"fontsize":10},
                  bins = bins,
                  color = 'navy',
                  smooth=smooth,
                  fig=fig)
    
    # Add Galactic centre
    ra_gc = 266.41683
    dec_gc = -29.0078
    axes = np.array(fig.axes).reshape((2, 2))
    ax = axes[1, 0]  # RA vs Dec subplot
    ax.plot(ra_gc, dec_gc, 'x', color = 'red', markersize=6, markeredgewidth=1)

    plt.tight_layout()

    if save:
        plt.savefig(f"plots/main_plots/{save}.png", dpi=300, bbox_inches='tight')

    plt.show()    



def plot_density_2d(posterior_sample, bins=100, cmap='jet', log_scale=True, save=None):

    """
    Creates a 2D histogram of g_x vs g_y components of the glide vector.

    Parameters
    ----------
    posterior_sample : dict
        Posterior samples with 'theta' key.
    bins : int, optional
        Number of bins in each dimension.
    cmap : str, optional
        Colormap for the histogram.
    log_scale : bool, optional
        Whether to apply logarithmic color normalization.
    save : str or None, optional
        Path to save the image (without extension).

    Example
    -------
    >>> plot_density_2d(posterior_sample, bins=200, save="gx_gy_density")
    """


    plt.figure(figsize=(6, 5))

    ps = posterior_sample
    C1 = np.sqrt(4*np.pi/3)/1000

    gx = -ps['theta'][:,4]/C1
    gy = ps['theta'][:,5]/C1

    # 2D histogram of gx vs gy
    plt.hist2d(
        gx, gy, bins=bins, cmap=cmap,
        norm=LogNorm() if log_scale else None
    )

    plt.xlabel(r"$g_x$ ($\mu$as/yr)")
    plt.ylabel(r"$g_y$ ($\mu$as/yr)")
    plt.colorbar(label='Sample count')
    plt.grid(False)
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    if save:
        plt.savefig(f"plots/main_plots/{save}.png", dpi=300, bbox_inches='tight')
    plt.show()


def mollweide_proj_galactic(posterior_sample, bins=300, save = None):

    """
    Projects posterior samples of the glide direction onto a Mollweide sky map in Galactic coordinates.

    Parameters
    ----------
    posterior_sample : dict
        Posterior samples with 'theta' key.
    bins : int, optional
        Number of bins for the 2D histogram (default is 300).
    save : str or None, optional
        If provided, saves the plot under 'plots/main_plots/{save}.png'.

    Example
    -------
    >>> mollweide_proj_galactic(posterior_sample, bins=300, save="glide_mollweide")
    """


    ps = posterior_sample

    # Convert s_10, s_11r, s_11i to Cartesian components
    C0 = np.sqrt(8 * np.pi / 3)
    C1 = np.sqrt(4 * np.pi / 3)

    g_x = -ps[:, 4] / C1
    g_y = ps[:, 5] / C1
    g_z = ps[:, 1] / C0

    # Normalise
    norm = np.sqrt(g_x**2 + g_y**2 + g_z**2)
    gx_u, gy_u, gz_u = g_x / norm, g_y / norm, g_z / norm

    # Convert to Galactic coordinates
    coords = SkyCoord(x=gx_u, y=gy_u, z=gz_u, representation_type='cartesian', frame='icrs')
    gal = coords.galactic
    l = gal.l.wrap_at(180*u.deg).radian  # [-pi, pi]
    b = gal.b.radian

    # Compute histogram manually
    lon_bins = np.linspace(-np.pi, np.pi, bins + 1)
    lat_bins = np.linspace(-np.pi/2, np.pi/2, bins + 1)
    hist, lon_edges, lat_edges = np.histogram2d(l, b, bins=[lon_bins, lat_bins])

    # Plot
    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.add_subplot(111, projection='mollweide')

    # pcolormesh expects 2D bins (edges), so use meshgrid
    Lon, Lat = np.meshgrid(lon_edges, lat_edges)
    mesh = ax.pcolormesh(Lon, Lat, hist.T, cmap='jet', norm=LogNorm())

    plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.07, label='Sample count')

    ax.grid(True)
    ax.set_xlabel("Galactic Longitude (rad)")
    ax.set_ylabel("Galactic Latitude (rad)")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()
    if save:
        plt.savefig(f"plots/main_plots/{save}.png", dpi=300, bbox_inches='tight')
    plt.show()
