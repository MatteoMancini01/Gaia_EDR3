# Saving HCM Posterior Samples \& Diagnostic
---
This directory is designed to store the saved HMC posteriors and their respective diagnostics, hence the choice of the respective sub-directories:
-	[`diagnostic_hmc`](diagnostic_hmc), store results from `numpyro.diagnostic.summary`
-	[`posterior_samples`](posterior_samples), store results after running HMC algorithm, particularly `mcmc.get_sammples()`.
This will help us to perform further analysis without running the sampling algorithm repeatedly.   
