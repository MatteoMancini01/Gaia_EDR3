# Project Overview

# Downloading Data

1. <b>Access the Gaia Archive:</b>
     - Go to the Gaia Archive: (https://gea.esac.esa.int/archive/)[https://gea.esac.esa.int/archive/]
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


