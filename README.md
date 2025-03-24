# Deep-learning-to-emulate-X-ray-spectral-models

## Description

Observing the X-ray Universe involves acquiring spectra over an energy band, covering from 0.1 to 10 keV. The spectra result from the convolution of the spectrum emitted by an astrophysical source, such as an accreting black hole, by the response of the instrument. X-ray spectrometers simultaneously provide images, where each pixel of the sky contains a spectrum. This will be the case for the X-ray Integral Field Unit of the future European X-ray observatory Athena (Barret et al. 2023, Experimental Astronomy). These spectra are then fitted by multi-parametric models. Model parameters and their uncertainties are obtained by Bayesian inference techniques (Dupourqué et al., 2024, A&A, JAXspec) or by simulation based inference involving deep learning (Barret & Dupourqué 2024, A&A, SIXSA). These parameters are then interpreted to constrain the properties of observed objects, such as the spin of black holes, or the abundances of hot gas in galaxy clusters. Libraries of emission models are available, either analytically or in the form of tables.

The aim of this project is to develop a neural network architecture to emulate any of these models.

The second step of this project is to couple the emulator to JAXspec (Dupourqué et al., 2024, A&A). JAXspec is a purely Python library for statistical inference on X-ray spectra. It enables spectral models to be built simply by combining components, and fitted to one or more observed spectra using Bayesian approaches. As JAXspec is written using JAX, all inference problems are compiled “just-in-time” and can be run on CPU or GPU.
