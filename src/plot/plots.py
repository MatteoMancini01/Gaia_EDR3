import corner
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm
from astropy.coordinates import SkyCoord
from astropy import units as u

def corner_plot(posterior_sample, index = [1, 4, 5]):

    samples_array = np.column_stack([posterior_sample['theta'][:, param] for param in index])
    plt.figure(figsize=(6.5, 4))  
    corner.corner(samples_array,
                        labels = [r"$s_{10}$", r"$s_{11}^{R}$", r"$s_{11}^i$"],
                        quantiles=[0.15, 0.5, 0.85], 
                        show_titles=True,
                        title_kwargs={"fontsize":14},
                        label_kwargs={"fontsize":14},
                        bins=30,
                        smooth=1)
    plt.show()

def corner_plot_eq_comp(posterior_sample, index = [1,4,5]):
    ps = posterior_sample

    # Define C0 and C1 for convertion to equatorial coordinates

    C0 = np.sqrt(8*np.pi/3)/1000
    C1 = np.sqrt(4*np.pi/3)/1000

    samples_array = np.column_stack([ps['theta'][:,1]/C0, -ps['theta'][:,4]/C1, ps['theta'][:,5]/C1])

    plt.figure(figsize=(6.5, 4))  
    corner.corner(samples_array, 
                    labels = [r"$g_z$", r"$g_x$", r"$g_y$"],
                    quantiles=[0.15, 0.5, 0.85], 
                    show_titles=True,
                    title_kwargs={"fontsize":14},
                    label_kwargs={"fontsize":14},
                    bins=30,
                    color="navy",
                    smooth=1)
    plt.show()

def plot_density_2d(posterior_sample, bins=100, cmap='jet', log_scale=True):
    plt.figure(figsize=(6, 5))

    ps = posterior_sample
    C1 = np.sqrt(4*np.pi/3)/1000

    gx = -ps['theta'][:,4]/C1
    gy = ps['theta'][:,5]/C1

    # 2D histogram of gx vs gy
    counts, xedges, yedges, img = plt.hist2d(
        gx, gy, bins=bins, cmap=cmap,
        norm=LogNorm() if log_scale else None
    )

    plt.xlabel(r"$g_x$ ($\mu$as/yr)")
    plt.ylabel(r"$g_y$ ($\mu$as/yr)")
    plt.colorbar(label='Sample count')
    #plt.title("Posterior Density of $g_x$ vs $g_y$", fontsize=15)
    plt.grid(False)
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    plt.show()

def mollweide_proj_galactic(posterior_sample, bins=300):
    ps = posterior_sample

    # Convert s_10, s_11r, s_11i to Cartesian components
    C0 = np.sqrt(8 * np.pi / 3)
    C1 = np.sqrt(4 * np.pi / 3)

    g_x = -ps['theta'][:, 4] / C1
    g_y = ps['theta'][:, 5] / C1
    g_z = ps['theta'][:, 1] / C0

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
    plt.show()
