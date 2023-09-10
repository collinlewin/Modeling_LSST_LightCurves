# Modeling LSST Light Curves of AGN
## Background
The Legacy Survey of Space and Time (LSST; set to begin in 2024) will revolutionize our understanding of supermassive black holes at the center of galaxies (i.e. Active Galactic Nuclei; AGN), among many, many facets of the universe. LSST will observe millions of these objects over ten years in six filters (wavelength ranges/wave bands) at optical wavelengths. How frequently each field of the sky will be observed is currently being decided. 

## Methodology
In this project, we modeled the simulated LSST observations of AGN from Sheng et al. (2022) using Gaussian processes. We aimed to test, for each combination of filter and cadence option, whether Gaussian process regression will be a reliable tool for estimating the power spectral density (PSD) of AGN for this decade-long survey. These time series were simulated using a damped random walk model commonly used to model AGN variability, which gives rise to a power law shape in the PSD with a high-frequency slope of -2. We model the dataset for each cadence and filter, and compute the PSD by drawing equally sampled realizations/samples from the Gaussian process posterior. Examples of these realizations for each cadence option (in the u-band) can be found in u_lc_plot.png. The PSDs are then each fit with a power law to assess how well the true underlying PSD properties can be estimated using this technique. 

![image](https://github.com/collinlewin/Modeling_LSST_LightCurves/assets/28280691/c64bd596-5132-4095-aaa0-08d93d809a13)

## Results
Out of the four tested kernel function forms (which have shown previous success in modeling AGN variability; see Wilkins+2019, Lewin+2021,2022), the Matern-1/2 kernel performs the best for modeling the observed variability in all filters and cadence options based on the negative log marginal likelihood (NLML; see the Table in nlml_results.png). We find that the modeling successfully preserves the underlying properties of the variability quantified by the PSD: the true (-2) slope is recovered within 1 std. dev. from the mean (in the distribution of the 400 best-fit PSD slopes) in all filters and cadence options (see the plot in slope_dists.png). 

![image](https://github.com/collinlewin/Modeling_LSST_LightCurves/assets/28280691/88065c5c-3b1b-4319-a55a-0cfaeb17a725)
