"""
Emulator of a diskbb spectrum from a certain temperature at inner disk radius (keV).
diskbb model represent the spectrum from an accretion disk consisting of multiple blackbody components.

For further informations on the model : https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node165.html

Usage :
>>> spec = diskbb(7.5)
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pcax  

# === File paths ===
data_path = Path("/Users/xifumacbook/Documents/Codes/Spectres/diskbb/diskbb06-001-50.npz")
model_path = Path("/Users/xifumacbook/Documents/Codes/Émulateur/diskbb/diskbb_model.keras")
pca_path = Path("/Users/xifumacbook/Documents/Codes/Émulateur/diskbb/diskbb_pca.pkl")
spline_path = data_path.with_suffix(".pkl")

# === Load data ===
spectra = np.log1p(np.load(data_path)["simulations"])  # Apply log1p to compress dynamic range
theta = np.load(data_path)["theta"]
temperatures = theta[:, 0].reshape(-1, 1)  # Extract temperatures as input parameter

# Load PCA state
with open(pca_path, "rb") as f:
    pca_state = pickle.load(f)

# Load temperature → normalization spline
with open(spline_path, "rb") as f:
    spline_N = pickle.load(f)

# Load trained model
model = load_model(model_path)

# Fit scaler on input data
scaler = StandardScaler()
spectra_scaled = scaler.fit_transform(spectra)

# === Physical normalization function ===
def get_norm_spline(T):
    """
    Return the physical normalization factor for temperature T using the spline.

    Parameters
    ----------
    T : float
        Temperature in keV.

    Returns
    -------
    float
        Normalization factor.
    """
    return 10 ** spline_N(np.log10(T))


# === Main reconstruction function ===
def diskbb_spectra(T_keV):
    """
    Reconstruct a normalized X-ray spectrum for a given diskbb temperature (keV).

    Parameters
    ----------
    T_keV : float
        Input temperature in keV.

    Returns
    -------
    np.ndarray
        Reconstructed spectrum (flux vs energy).
    """
    # Predict PCA-reduced spectrum from the model
    pred_pca = model.predict(np.array([[T_keV]]), verbose=0)

    # Recover full PCA vector
    pred_scaled = pcax.recover(pca_state, pred_pca)

    # Inverse transform and undo log1p + normalize
    pred_spec = np.expm1(scaler.inverse_transform(pred_scaled).flatten())
    pred_spec /= get_norm_spline(T_keV)

    return pred_spec