
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

def get_vsh_index_map(lmax):
    """
    Creates a mapping from index to VSH coefficient label based on model_vsh() indexing logic.

    Args:
        lmax (int): Maximum degree of VSH expansion.

    Returns:
        dict: Mapping from index (int) to string name of coefficient (e.g., 't_10', 's_11r')
    """
    index_map = {}
    idx = 0
    for l in range(1, lmax + 1):
        for m in range(0, l + 1):
            if m == 0:
                index_map[idx] = f"t_{l}{m}"
                index_map[idx + 1] = f"s_{l}{m}"
                idx += 2
            else:
                index_map[idx]     = f"t_{l}{m}r"
                index_map[idx + 1] = f"t_{l}{m}i"
                index_map[idx + 2] = f"s_{l}{m}r"
                index_map[idx + 3] = f"s_{l}{m}i"
                idx += 4
    return index_map

def vsh_minuit_limits(lmax, t_bound=0.01, s_bound=0.01):

    """
    Generates a dictionary of parameter bounds for use with Minuit optimisers based on the VSH expansion up to degree `lmax`.

    Args:
        lmax (int): Maximum VSH degree to include in the parameterisation.
        t_bound (float, optional): Symmetric bound for toroidal (T_lm) coefficients. Default is +/-0.01 mas/yr.
        s_bound (float, optional): Symmetric bound for spheroidal (S_lm) coefficients. Default is +/-0.01 mas/yr.

    Returns:
        dict: A mapping from parameter names (e.g., "x0", "x1", ...) to (lower, upper) bound tuples for use in Minuit.

    Notes:
        - For each (l, m) mode:
            - If m = 0: adds 2 parameters (t_lm, s_lm).
            - If m > 0: adds 4 parameters (Re/Im parts of t_lm and s_lm).
        - Total parameter count = 2 * lmax * (lmax + 2), matching the length of the `theta` vector.
        - Intended for use in constrained VSH fitting pipelines, such as Minuit-based optimisation.
    
    Example:
        >>> from src.models.vsh_model import vsh_minuit_limits
        >>> limits = vsh_minuit_limits(lmax=2, t_bound=0.01, s_bound=0.0085)

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


def cov_matrix_hmc(posterior_sample, indices=None):
    theta_samples = np.array(posterior_sample)
    cov_matrix = np.cov(theta_samples, rowvar=False)
    
    if indices is not None:
        cov_matrix = cov_matrix[np.ix_(indices, indices)]

    return cov_matrix

def rho_matrix(cov_matrix):

    stddevs = np.sqrt(np.diag(cov_matrix))  # shape (3,)
    correlation_matrix = cov_matrix/np.outer(stddevs, stddevs)

    return correlation_matrix

def vsh_vector_summary(params, cov_matrix, kind="glide"):
    # kind: "glide" or "rotation"
    C0 = np.sqrt(8*np.pi/3)
    C1 = np.sqrt(4*np.pi/3)

    s10, s11r, s11i = params[0], params[1], params[2]

    vx = -s11r/C1
    vy = s11i/C1
    vz = s10/C0

    v_vec = np.array([vx, vy, vz])
    v_mag = np.linalg.norm(v_vec)

    Jac = np.array([[-1/C1, 0, 0],  
                     [0, 1/C1, 0],
                     [0, 0, 1/C0]])
    
    Sigma_v = Jac @ cov_matrix @ Jac
    sigma_v = np.sqrt(np.diag(Sigma_v))
    sigma_v_mag = np.sqrt((v_vec.T @ Sigma_v @ v_vec) / v_mag**2)
    corr_matrix = rho_matrix(Sigma_v)

    label = "|g|" if kind == "glide" else "|ω|"
    vector_label = "g" if kind == "glide" else "ω"

    return {
        f"{label} (μas/yr)": v_mag*1000,
        f"{vector_label} (μas/yr)": v_vec*1000,
        f"|sigma_{vector_label}| (μas/yr)": sigma_v_mag*1000,
        f"sigma_{vector_label} (μas/yr)": sigma_v*1000,
        f"Corr_{vector_label}x_{vector_label}y": corr_matrix[0, 1],
        f"Corr_{vector_label}x_{vector_label}z": corr_matrix[0, 2],
        f"Corr_{vector_label}y_{vector_label}z": corr_matrix[1, 2],
    }, v_vec, Sigma_v, corr_matrix

def ra_dec_summary(param, covariance):
    
    vx, vy, vz = param[0], param[1], param[2]

    r0 = vx**2 + vy**2 + vz**2
    r1 = vx**2 + vy**2

    ra_rad = np.arctan2(vy,vx) % (2*np.pi)
    dec_rad = np.arcsin(vz/np.sqrt(r0))

    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.rad2deg(dec_rad)

    Jac = np.array([[-vy/r1, vx/r1, 0],
                     [-vx*vy/(r0*np.sqrt(r1)), -vy*vz/(r0*np.sqrt(r1)), np.sqrt(r1)/r0]])
    
    Sigma_ra_dec = Jac @ covariance @ Jac.T

    sigma_ra_deg = np.rad2deg(np.sqrt(Sigma_ra_dec[0][0]))
    sigma_dec_deg = np.rad2deg(np.sqrt(Sigma_ra_dec[1][1]))
    Cov_ra_dec_deg = np.rad2deg(Sigma_ra_dec[0][1])
    Corr_ra_dec = Cov_ra_dec_deg/sigma_dec_deg*sigma_ra_deg

    return {'RA (deg)': ra_deg,
            'Sigma_RA (deg)': sigma_ra_deg,
            'Dec (deg)': dec_deg,
            'Sigma_Dec (deg)': sigma_dec_deg,
            'Corr_RA_dec': Corr_ra_dec}

def get_icrs_to_galactic_rotation():
    # From Reid & Brunthaler (2004), used by Astropy internally (see docs)
    return np.array([
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [ 0.4941094279, -0.4448296300,  0.7469822445],
        [-0.8676661490, -0.1980763734,  0.4559837762]
    ])


def vsh_vector_summary_galactic(v_vec_eq, Sigma_eq, kind="glide"):
    # Rotate vector and covariance
    R = get_icrs_to_galactic_rotation()
    v_vec_gal = R @ v_vec_eq
    Sigma_gal = R @ Sigma_eq @ R.T
    sigma_gal = np.sqrt(np.diag(Sigma_gal))
    v_mag_gal = np.linalg.norm(v_vec_gal)
    sigma_v_mag = np.sqrt((v_vec_gal.T @ Sigma_gal @ v_vec_gal) / v_mag_gal**2)
    corr_matrix = rho_matrix(Sigma_gal)

    label = "|g|_gal" if kind == "glide" else "|ω|_gal"
    vector_label = "g_gal" if kind == "glide" else "ω_gal"

    return {
        f"{label} (μas/yr)": v_mag_gal*1000,
        f"{vector_label} (μas/yr)": v_vec_gal*1000,
        f"|sigma_{vector_label}| (μas/yr)": sigma_v_mag*1000,
        f"sigma_{vector_label} (μas/yr)": sigma_gal*1000,
        f"Corr_{vector_label}x_{vector_label}y": corr_matrix[0, 1],
        f"Corr_{vector_label}x_{vector_label}z": corr_matrix[0, 2],
        f"Corr_{vector_label}y_{vector_label}z": corr_matrix[1, 2],
    }, v_vec_gal, Sigma_gal, corr_matrix

def lb_summary(v_gal, Sigma_gal):
    """
    Computes Galactic longitude and latitude + uncertainties and correlation
    from Cartesian Galactic vector and covariance.

    Parameters:
        v_gal: array-like, shape (3,) — glide vector in galactic Cartesian coords (μas/yr)
        Sigma_gal: array-like, shape (3, 3) — covariance in galactic frame

    Returns:
        dict with l, b and their uncertainties and correlation
        
    """

    vx, vy, vz = v_gal

    r0 = vx**2 + vy**2 + vz**2
    r1 = vx**2 + vy**2

    l_rad = np.arctan2(vy, vx) % (2*np.pi)
    b_rad = np.arcsin(vz / np.sqrt(r0))

    l_deg = np.rad2deg(l_rad)
    b_deg = np.rad2deg(b_rad)

    # Jacobian for [l, b] w.r.t. [vx, vy, vz]
    J = np.array([
        [-vy / r1, vx / r1, 0],
        [-vx * vz / (r0 * np.sqrt(r1)), -vy * vz / (r0 * np.sqrt(r1)), np.sqrt(r1) / r0]
    ])

    Sigma_lb = J @ Sigma_gal @ J.T

    sigma_l_deg = np.rad2deg(np.sqrt(Sigma_lb[0, 0]))
    sigma_b_deg = np.rad2deg(np.sqrt(Sigma_lb[1, 1]))
    Cov_lb = np.rad2deg(Sigma_lb[0, 1])  # cross-covariance in degrees²
    Corr_lb = Cov_lb / (sigma_l_deg * sigma_b_deg)

    return {
        "l (deg)": l_deg,
        "Sigma_l (deg)": sigma_l_deg,
        "b (deg)": b_deg,
        "Sigma_b (deg)": sigma_b_deg,
        "Corr_l_b": Corr_lb
    }

def print_summary(dictionary, title=None, indent=2, precision=4):
    """
    Pretty-print a summary dictionary with optional title and formatting.

    Args:
        dictionary (dict): Dictionary of key-value pairs.
        title (str, optional): Optional title header.
        indent (int): Spaces for indenting each entry.
        precision (int): Number of decimal places for floats.
    """
    space = " " * indent
    if title:
        print(f"{title}")
        print("-" * len(title))

    for key, value in dictionary.items():
        if isinstance(value, float):
            print(f"{space}{key:<20}: {value:.{precision}f}")
        elif isinstance(value, (np.float32, np.float64)):
            print(f"{space}{key:<20}: {float(value):.{precision}f}")
        elif isinstance(value, int):
            print(f"{space}{key:<20}: {value:d}")
        else:
            print(f"{space}{key:<20}: {value}")



def config_data(df):
    
    # Preparing dataset
    ra_rad = np.deg2rad(np.array(df["ra"].values))
    dec_rad = np.deg2rad(np.array(df["dec"].values))

    # Group as [ra, dec] shape = (2, N)
    angles = np.stack([ra_rad, dec_rad])

    # Prepare observed proper motions
    pmra = np.array(df["pmra"].values)
    pmdec = np.array(df["pmdec"].values)

    # Group as [pmra, pmdec] shape = (2, N)
    obs = np.stack([pmra, pmdec])

    # Prepare error
    pmra_error = np.array(df["pmra_error"].values)
    pmdec_error = np.array(df["pmdec_error"].values)
    corr = np.array(df["pmra_pmdec_corr"].values)

    # Group as [pmra_error, pmdec_error, pmra_pmdec_corr] shape = (3, N)
    error = np.stack([pmra_error, pmdec_error, corr])

    return angles, obs, error 


def chi2_red(minuit_result, lmax, n_data = None):

    if n_data == None:
        n_data = 1215942
    else:
        n_data = n_data

    chi2 = minuit_result.fval
    n_param = 2*lmax*(lmax+1)
    ndof = n_data*2 - n_param
    chi2_red = chi2 / ndof

    print(f'Goodness of fit χ^2_red =', chi2_red)

def cov_ts(hessian, index):
    h = hessian
    i = index
    cov = np.array([[h[i[0]][i[0]], h[i[0]][i[1]], h[i[0]][i[2]]],
                     [h[i[1]][i[0]], h[i[1]][i[1]], h[i[1]][i[2]]],
                     [h[i[2]][i[0]], h[i[2]][i[1]], h[i[2]][i[2]]]])
    return cov