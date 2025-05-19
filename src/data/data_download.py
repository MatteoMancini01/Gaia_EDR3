from astroquery.gaia import Gaia

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
result.write("qso_full_data.csv", format="csv", overwrite=True) # overwrite = True will allow to replace existing data when redownloaded.