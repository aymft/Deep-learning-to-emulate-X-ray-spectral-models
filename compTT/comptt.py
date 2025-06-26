"""
Module for reconstructing a compTT X-ray spectrum from physical parameters

compTT describes Comptonization of soft photons in a hot plasma.

For further informations on the model : https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node159.html

Usage example:
>>> spectrum = comptt_spectra(T0=0.5, kT=2.0, tau=1.2)
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pcax

# === File paths ===
model_path = Path("/Users/xifumacbook/Documents/Codes/Émulateur/compTT/comptt_model.keras")
pca_path = Path("/Users/xifumacbook/Documents/Codes/Émulateur/compTT/comptt_pca.pkl")
data_path = Path("/Users/xifumacbook/Documents/Codes/Spectres/compTT/comptt_Didier_approved.npz")

# === Load data ===
data = np.load(data_path)
spectra = np.log1p(data["simulations"])  # Apply log1p for dynamic range compression
theta_raw = data["theta"]

# Remove unused parameters (columns [0], [4], and [-1])
theta_cleaned = np.delete(theta_raw, [0, 4, -1], axis=1)

# === Load model and PCA state ===
model = load_model(model_path)

with open(pca_path, "rb") as f:
    pca_state = pickle.load(f)

# === Fit MinMaxScaler on the full spectra set ===
scaler_spectra = MinMaxScaler()
spectra_scaled = scaler_spectra.fit_transform(spectra)

# === Main function ===
def comptt_spectra(T0, kT, tau):
    """
    Reconstruct a compTT spectrum given input physical parameters.

    Parameters
    ----------
    T0 : float
        Input soft photon (Wien) temperature (keV)
    kT : float
        Plasma temperature (keV)
    tau : float
        Plasma optical depth (dimensionless)

    Returns
    -------
    np.ndarray
        Reconstructed spectrum (flux vs energy).
    """
    # Build the input vector in correct order
    input_theta = np.array([[T0, kT, tau]])

    # Predict PCA-compressed spectrum
    y_pred_pca = model.predict(input_theta, verbose=0)

    # Recover full spectrum: PCA inverse + scaler inverse + expm1
    y_scaled = pcax.recover(pca_state, y_pred_pca)
    y_log = scaler_spectra.inverse_transform(y_scaled)
    spectrum = np.expm1(y_log).flatten()

    return spectrum
