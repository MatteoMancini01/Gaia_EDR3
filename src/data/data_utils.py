import pandas as pd
import healpy as hp
import numpy as np

def load_qso_dataframe(path="csv_files/qso_full_data.csv", astrometric_flag=31):
    """
    Loads the QSO dataset as a pandas DataFrame (stored in CPU RAM).

    Args:
        path (str): Path to CSV file
        astrometric_flag (int): Filter value for astrometric_params_solved

    Returns:
        pd.DataFrame: Filtered QSO data
    """
    df = pd.read_csv(path)
    df_filtered = df[df["astrometric_params_solved"] == astrometric_flag]
    return df_filtered

def load_filtered_qso_df(path = "csv_files/filtered_qso_data.csv"):

    df = pd.read_csv(path)
    return df


def bin_qso_data_healpix(df, nside=32):
    """
    Bins Gaia QSO data into HEALPix pixels and averages proper motions and errors.

    Args:
        df (pd.DataFrame): Gaia QSO data with columns: 'ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'pmra_pmdec_corr'
        nside (int): HEALPix NSIDE value (e.g., 32 or 64)

    Returns:
        pd.DataFrame: One row per pixel with averaged values
    """
    theta = 0.5*np.pi - np.deg2rad(df["dec"].values)
    phi = np.deg2rad(df["ra"].values)


    pix = hp.ang2pix(nside, theta, phi, nest=False)
    df["healpix"] = pix

    grouped = df.groupby("healpix")
    binned_df = grouped[[
        "ra", "dec", "pmra", "pmdec", "pmra_error", "pmdec_error", "pmra_pmdec_corr"
    ]].agg({
        "ra": "mean",
        "dec": "mean",
        "pmra": "mean",
        "pmdec": "mean",
        "pmra_error": lambda x: np.sqrt(np.mean(x**2)),
        "pmdec_error": lambda x: np.sqrt(np.mean(x**2)),
        "pmra_pmdec_corr": "mean"
    }).reset_index()

    return binned_df
