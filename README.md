# Project Overview

key publication [link](https://www.aanda.org/articles/aa/full_html/2021/05/aa39734-20/aa39734-20.html)

reference for vector spherical harmonics [link](https://www.aanda.org/articles/aa/pdf/2012/11/aa19927-12.pdf)

# Downloading Data

1. <b>Access the Gaia Archive:</b>
     - Go to the Gaia Archive: [https://gea.esac.esa.int/archive/](https://gea.esac.esa.int/archive/)
     - This is the official ESA Gaia Archive, where you can query and download the data.

2. <b>Query for QSO-like Objects:</b>
     - The QSO-like objects in Gaia EDR3 are provided in the agn_cross_id table.
     - To download the full dataset run the following command on your terminal:
          ```bash
          wget -r -np -nH --cut-dirs=3 -A "*.csv.gz" https://cdn.gea.esac.esa.int/Gaia/gedr3/auxiliary/agn_cross_id/
          ```
        This allows you to download all the ```.cvs.gz``` files.

3. <b>Checking Data</b>
     - Extract the files:
       If the files are compressed in a (```.gz```) format, you can extract them with:
       ```bash
       gunzip agn_cross_id/*.csv.gz
       ```
       (<b>Note:</b> make sure you are in the right directory!)

     - Load the data in Python:

4. <b>Downloading full dataset using Python</b>
The above procedure does not allow the download of the full dataset. Instead run the following python code on you notebook. Note you need the Python package [```astroquery```](https://astroquery.readthedocs.io/en/latest/).

Install ```astroquery``` with ```pip install astroquery```, one installed run the following code on your notebook. 
```python

from astroquery.gaia import Gaia
import pandas as pd

# Query to get the astrometric properties of QSO-like objects
query = """
SELECT 
    agn.source_id, 
    gs.ra, gs.dec, 
    gs.pmra, gs.pmdec, 
    gs.parallax, gs.parallax_error, 
    gs.ruwe, gs.phot_g_mean_mag 
FROM gaiadr3.agn_cross_id AS agn
JOIN gaiadr3.gaia_source AS gs 
ON agn.source_id = gs.source_id
WHERE gs.parallax < 5 * gs.parallax_error  -- Remove potential stars
AND gs.ruwe < 1.4  -- Ensure good astrometric quality
AND gs.phot_g_mean_mag < 21  -- Bright enough for good measurements
"""

# Launch query and download data
job = Gaia.launch_job_async(query)
result = job.get_results()

# Save as CSV
result.write("qso_full_data.csv", format="csv", overwrite=True)

# Load into Pandas
df = pd.read_csv("qso_full_data.csv")
print(df.head())  # Check the data

```
This will allow you to store the data in your directory, in the file ```qso_full_data.csv```. 