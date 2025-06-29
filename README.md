#  Project Overview :mag: :memo:

key publication [link](https://www.aanda.org/articles/aa/full_html/2021/05/aa39734-20/aa39734-20.html)

reference for vector spherical harmonics [link](https://www.aanda.org/articles/aa/pdf/2012/11/aa19927-12.pdf)

Useful [link](https://irsa.ipac.caltech.edu/data/Gaia/dr3/gaia_dr3_source_colDescriptions.html) (checking data units)
---

---
## Getting started :rocket:
---
### Python Virtual Enviroments :snake: :test_tube:

1. Create a Virtual Environment

   Run the following command to create your virtual environment

   ``` bash
    python -m venv <your_env>

- If the above command fails, please try:
   ```bash
   python3 -m venv <your_env>

Replace `<your_env>` with your preferred environment name, e.g. `gaia_venv`.

2. Activate your virtual environment

  Activate your virtual environment with:
   ```bash
    source <your_env>/bin/activate
   ```
  Deactivate your environment with:
   ```bash
    deactivate
   ```
3. To install all the required libraries please run the command:
   ```bash
   pip install -r requirements.txt
   ```
---

### Bonus Tip :gift:

Most functions in this repository are coded with [JAX](https://docs.jax.dev/en/latest/index.html), a library for numerical computation resembling most properties of [NumPy](https://numpy.org/), with automatic differentiation (hence natural choice for coding VSH functions) including [JIT](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) compilation to speed up computations. 
Like any other Python library, JAX can be installed for CPU on Linux, Windows and macOS. Additionally, if you machine has access to a NVIDIA GPU, it is recommended that you install JAX for GPU with:
  ```bash
  pip install -U "jax[cuda12]"
  ```
If you have a GPU, I strongly recommend installing JAX using the command above, as it will significantly speed up computations! To verify that your GPU is active with JAX run the following command:
  ```python
  import jax
  print(jax.devices())
  ```
If the output shows:
 ```bash
[CudaDevice(id=0)]
 ```
Then your GPU is active and ready to compute with JAX. For more infromation please see [source](https://docs.jax.dev/en/latest/quickstart.html).
---

### Conda Environments :snake: :test_tube:

1. Create a Conda Environment
   Run the following command to create your Conda environment

    ```bash
    conda env create -f environment.yml
    ```

All the required libraries will be automatically installed.

2. Activate your Conda envorpnment

    ```bash
    conda activate m2_venv
    ```

   To deactivate: 
   
    ```bash
    conda deactivate
    ```
---

## Download Data :ringed_planet: :telescope: :floppy_disk:

1. <b>Access the Gaia Archive:</b> :card_index_dividers:
     - Go to the Gaia Archive: [https://gea.esac.esa.int/archive/](https://gea.esac.esa.int/archive/)
     - This is the official ESA Gaia Archive, where you can query and download the data.

2. <b>Query for QSO-like Objects:</b> :sparkles: 
     - The QSO-like objects in Gaia EDR3 are provided in the agn_cross_id table.
     - To download the full dataset run the following command on your terminal:
          ```bash
          wget -r -np -nH --cut-dirs=3 -A "*.csv.gz" https://cdn.gea.esac.esa.int/Gaia/gedr3/auxiliary/agn_cross_id/
          ```
        This allows you to download all the ```.cvs.gz``` files.

3. <b>Checking Data</b> :hammer_and_wrench:
     - Extract the files:
       If the files are compressed in a (```.gz```) format, you can extract them with:
       ```bash
       gunzip agn_cross_id/*.csv.gz
       ```
       (<b>Note:</b> make sure you are in the right directory!)

     - Load the data in Python:

4. <b>Downloading full dataset using Python</b> :arrow_down: :inbox_tray:

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
    gs.ruwe, gs.phot_g_mean_mag,
    gs.nu_eff_used_in_astrometry,
    gs.pmra_error,
    gs.pmdec_error,
    gs.pmra_pmdec_corr,
    gs.astrometric_params_solved
FROM gaiadr3.agn_cross_id AS agn
JOIN gaiadr3.gaia_source AS gs 
ON agn.source_id = gs.source_id

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