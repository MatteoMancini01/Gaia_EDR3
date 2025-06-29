# Saving HCM Posterior Samples \& Diagnostic
---
This directory is designed to store the saved HMC posteriors and their respective diagnostics, hence the choice of the respective sub-directories:
-	[`diagnostic_hmc`](diagnostic_hmc), store results from `numpyro.diagnostic.summary`
-	[`posterior_samples`](posterior_samples), store results after running HMC algorithm, particularly `mcmc.get_sammples()`.
This will help us to perform further analysis without running the sampling algorithm repeatedly.   
- [`main_results`](main_results), store posterior distribution and respective diagnostic for our main results, with lmax=3, 8 chains, 5000 samples per chain.
- [`extension`], store posterior samples and respective diagnostics, for analysis on full dataset  ([full_data](extension\full_data)) and two different clipping values kappa = 2 ([mask2](extension\mask2)), kappa = 4 ([mask4](extension\mask4)). See notebook [`extension.ipynb`](../extension.ipynb).
