
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
    Cov_ra_dec_deg = np.rad2deg(Sigma_ra_dec[0][0])
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


    

def spheroidal_vector_summary(parameters, variances, index = np.array([1,4,5])):

    """
    Summarizes the spheroidal (glide) vector field components as a Cartesian vector and corresponding sky coordinates.

    Args:
        s10 (float): Coefficient for the m = 0 spheroidal harmonic (S_10).
        s11r (float): Real part of the m = 1 spheroidal coefficient (Re(S_11)).
        s11i (float): Imaginary part of the m = 1 spheroidal coefficient (Im(S_11)).

    Returns:
        dict: A dictionary containing:
            - `"G_vector (mas/yr)"`: Cartesian glide vector [G1, G2, G3] in mas/yr.
            - `"Magnitude (μas/yr)"`: Norm of the vector in μas/yr.
            - `"RA (deg)"`: Right ascension of the vector direction in degrees.
            - `"Dec (deg)"`: Declination of the vector direction in degrees.

    Notes:
        - Normalisation constants (C0 and C1) follow the VSH formalism used in the literature (e.g., Gaia Collaboration 2021).
        - The resulting vector points in the direction of apparent systematic motion due to acceleration.
        - Returned coordinates are suitable for visualization or physical interpretation.

    Example:
        >>> from src.models.vsh_model import spheroidal_vector_summary
        >>> spheroidal_vector_summary(0.0002, 0.0001, 0.000253, 4e-5, 1e-6, 7e-6)
            
    """
    R = np.array([[-0.054875560, -0.8734370902, -0.4838350155],
                  [0.4941094279, -0.4448296300, 0.7469822445],
                  [-0.8676661490, -0.1980763734, 0.4559837762]])
    i = index
    p = parameters
    v = variances
    s10, s11r, s11i = p[i[0]], p[i[1]], p[i[2]] 
    v_s10, v_s11r, v_s11i = v[i[0]], v[i[1]], v[i[2]] 

    # Normalisation constants from VSH paper
    C0 = np.sqrt(8*np.pi/3)
    C1 = np.sqrt(4*np.pi/3)

    # Convert s_lm to vector components
    G3 = s10/C0
    G1 = -s11r/C1
    G2 = s11i/C1

    sigmaG1 = np.sqrt((1/C1)**2*v_s11r)
    sigmaG2 = np.sqrt((1/C1)**2*v_s11i)
    sigmaG3 = np.sqrt((1/C0)**2*v_s10)


    Sigma_eq = np.diag(np.array([sigmaG1**2, sigmaG2**2, sigmaG3**2]))
    Sigma_gal = R @ Sigma_eq @ R.T
    sigma_gal = np.sqrt(np.diag(Sigma_gal))

    # Magnitude
    G_mag = np.sqrt(G1**2 + G2**2 + G3**2)
    G_eq_vec = np.array([G1, G2, G3]) # equatorial 
    G_gal_vec = R @ G_eq_vec

    # Magnitude uncertainty
    
    sigma_mag = np.sqrt((G1**2*v_s11r + G2**2*v_s11i + G3**2*v_s10)/G_mag**2)

    dec = np.arcsin(G3/G_mag)
    ra = np.arctan2(G2, G1) % (2*np.pi)

    l = np.arctan2(G_gal_vec[1], G_gal_vec[0])%(2*np.pi)
    d = np.arcsin(G_gal_vec[2]/G_mag)


    den_ra = G1**2 + G2**2
    error_ra_rad = np.sqrt((G2/den_ra)**2*v_s11r + (G1/den_ra)**2*v_s11i)

    den_dec_xy = G_mag**3*np.sqrt(1 - (G3/G_mag)**2)
    den_dec_z = G_mag*np.sqrt(1 - (G3/G_mag)**2)
    d_x = -G1*G3/den_dec_xy
    d_y = -G2*G3/den_dec_z
    d_z = (1 - (G3/G_mag)**2)
    error_dec_ra = np.sqrt(d_x**2*v_s11r + d_y**2*v_s11i + d_z**2*v_s10)

    den_l = G_gal_vec[0]**2 + G_gal_vec[1]**2
    error_l_rad = np.sqrt((G_gal_vec[1] / den_l)**2 * Sigma_gal[0, 0] + 
                            (G_gal_vec[0] / den_l)**2 * Sigma_gal[1, 1]) # error radians l

    den_d_xy = G_mag**3*np.sqrt(1 - (G_gal_vec[2]/G_mag)**2)
    den_d_z = G_mag*np.sqrt(1 - (G_gal_vec[2]/G_mag)**2)
    d_gal_x = -G_gal_vec[0]*G_gal_vec[2]/den_d_xy
    d_gal_y = -G_gal_vec[1]*G_gal_vec[2]/den_d_z
    d_gal_z = (1 - (G_gal_vec[2]/G_mag)**2)
    error_d_rad = np.sqrt(d_gal_x**2 * Sigma_gal[0, 0] +
                            d_gal_y**2 * Sigma_gal[1, 1] +
                            d_gal_z**2 * Sigma_gal[2, 2])


    G_mag_uas = G_mag*1000  # mas/yr -> μas/yr
    sigma_mag_uas = sigma_mag*1000 # # mas/yr -> μas/yr
    ra_deg = np.rad2deg(ra)
    dec_deg = np.rad2deg(dec)
    error_ra_deg = np.rad2deg(error_ra_rad)
    error_dec_deg = np.rad2deg(error_dec_ra)

    l_deg = np.rad2deg(l)
    d_deg = np.rad2deg(d)
    error_l_deg = np.rad2deg(error_l_rad)
    error_d_deg = np.rad2deg(error_d_rad)
    
    print("Equatorial components:")
    print(f"G_vec = {G_eq_vec*1000} +/- {np.array([sigmaG1, sigmaG2, sigmaG3])*1000}(μas/yr)")
    print(f"Magnitude = {G_mag_uas} +/- {sigma_mag_uas} (μas/yr)")
    print(f"RA = {ra_deg} +/- {error_ra_deg} (deg)")
    print(f"Dec = {dec_deg} +/- {error_dec_deg} (deg)")
    print("")
    print("Galactic components:")
    print(f"G_vec = {G_gal_vec*1000} +/- {sigma_gal*1000}(μas/yr)")
    print(f"l = {l_deg} +/- {error_l_deg} (deg)")
    print(f"d = {d_deg} +/- {error_d_deg} (deg)")

def toroidal_vector_summary(parameters, variances, index = np.array([0,2,3])):

    """
    Summarizes the toroidal (rotation-like) vector field components as a Cartesian vector and equatorial direction.

    Args:
        t10 (float): Coefficient for the m = 0 toroidal harmonic (T_10).
        t11r (float): Real part of the m = 1 toroidal coefficient (Re(T_11)).
        t11i (float): Imaginary part of the m = 1 toroidal coefficient (Im(T_11)).

    Returns:
        dict: A dictionary containing:
            - `"R_vector (mas/yr)"`: Cartesian rotation vector [R1, R2, R3] in mas/yr.
            - `"Magnitude (μas/yr)"`: Norm of the vector in μas/yr.
            - `"RA (deg)"`: Right ascension of the vector direction in degrees.
            - `"Dec (deg)"`: Declination of the vector direction in degrees.

    Notes:
        - This is analogous to the spheroidal summary but applies to toroidal components (spin-like structure).
        - Used to interpret rotation patterns in VSH modeling, such as global frame spin or bulk rotation.
        - Coordinates and magnitude are scaled for readability in physical units.
    
    Example:
        >>> from src.models.vsh_model import toroidal_vector_summary
        >>> result = toroidal_vector_summary(0.0002, 0.0001, 0.000253)
        >>> for key, value in result.items():
        >>>     print(f"{key:25}: {value}")
    """
    i = index
    p = parameters
    v = variances
    t10, t11r, t11i = p[i[0]], p[i[1]], p[i[2]] 
    v_t10, v_t11r, v_t11i = v[i[0]], v[i[1]], v[i[2]] 

    C0 = np.sqrt(8*np.pi/3)
    C1 = np.sqrt(4*np.pi/3)

    R3 = t10/C0
    R1 = -t11r/C1
    R2 = -t11i/C1

    sigmaR1 = np.sqrt((1/C1)**2*v_t11r)
    sigmaR2 = np.sqrt((1/C1)**2*v_t11i)
    sigmaR3 = np.sqrt((1/C0)**2*v_t10)

    R_mag = np.sqrt(R1**2 + R2**2 + R3**2)

    sigma_mag = np.sqrt(R1**2*v_t11r + R2**2*v_t11i + R3**2*v_t10)/R_mag

    dec = np.arcsin(R3/R_mag)
    ra = np.arctan2(R2, R1) % (2*np.pi)

    R_mag_uas = R_mag*1000  # mas/yr -> μas/yr
    ra_deg = np.rad2deg(ra)
    dec_deg = np.rad2deg(dec)

    den_ra = R1**2 + R2**2
    error_ra_rad = np.sqrt((R2/den_ra)**2*v_t11r + (R1/den_ra)**2*v_t11i)


    den_dec_xy = R_mag**3*np.sqrt(1 - (R3/R_mag)**2)
    den_dec_z = R_mag*np.sqrt(1 - (R3/R_mag)**2)
    d_x = -R1*R3/den_dec_xy
    d_y = -R2*R3/den_dec_z
    d_z = (1 - (R3/R_mag)**2)
    error_dec_ra = np.sqrt(d_x**2*v_t11r + d_y**2*v_t11i + d_z**2*v_t10)

   
    R_mag_uas = R_mag*1000  # mas/yr -> μas/yr
    sigma_mag_uas = sigma_mag*1000 # # mas/yr -> μas/yr
    ra_deg = np.rad2deg(ra)
    dec_deg = np.rad2deg(dec)
    error_ra_deg = np.rad2deg(error_ra_rad)
    error_dec_deg = np.rad2deg(error_dec_ra)

    print(f"R_vec = {np.array([R1, R2, R3])*1000} +/- {np.array([sigmaR1, sigmaR2, sigmaR3])*1000}(μas/yr)")
    print(f"Magnitude = {R_mag_uas} +/- {sigma_mag_uas} (μas/yr)")
    print(f"RA = {ra_deg} +/- {error_ra_deg} (deg)")
    print(f"Dec = {dec_deg} +/- {error_dec_deg} (deg)")


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