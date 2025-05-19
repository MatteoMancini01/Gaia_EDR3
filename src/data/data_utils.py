import pandas as pd

def load_qso_dataframe(path="qso_full_data.csv", astrometric_flag=31):
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
