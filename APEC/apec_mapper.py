import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt

# 📌 Chargement des données
data_path = Path("/tmpdir/ferec/apec_simon_approved.npz")
data = np.load(data_path)
spectres = data["simulations"]  # shape (n_samples, n_bins)
theta = data["theta"]           # shape (n_samples, n_params)
theta = np.delete(theta, [1, 2, -1], axis=1)  # Suppression redshift, approx, norm

# 📌 Transformation log1p
spectres = np.log1p(spectres)

# 📌 Split train / val avant scaling
theta_train, theta_val, spectres_train, spectres_val = train_test_split(
    theta, spectres, test_size=0.1, random_state=42, shuffle=True
)

# 📌 Normalisation sans fuite
scaler_spectre = MinMaxScaler().fit(spectres_train)
spectres_train_scaled = scaler_spectre.transform(spectres_train)
spectres_val_scaled = scaler_spectre.transform(spectres_val)

# 📌 Autoencodeur
def build_autoencoder(input_dim, latent_dim, learning_rate=1e-4):
    encoder = Sequential([
        Dense(256, activation='gelu', input_shape=(input_dim,)),
        Dense(256, activation='gelu'),
        Dense(256, activation='gelu'),
        Dense(latent_dim, activation='linear', name='latent_space')
    ])

    decoder = Sequential([
        Dense(256, activation='gelu', input_shape=(latent_dim,)),
        Dense(256, activation='gelu'),
        Dense(256, activation='gelu'),
        Dense(input_dim, activation='linear')
    ])

    inputs = Input(shape=(input_dim,))
    latent = encoder(inputs)
    outputs = decoder(latent)

    autoencoder = Model(inputs, outputs)
    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=MeanSquaredError()
    )

    return autoencoder, encoder, decoder

input_dim = spectres_train_scaled.shape[1]
latent_dim = 20
autoencoder, encoder, decoder = build_autoencoder(input_dim, latent_dim)
autoencoder.summary()


# 📌 Entraînement autoencodeur
history_ae = autoencoder.fit(
    spectres_train_scaled, spectres_train_scaled,
    validation_data=(spectres_val_scaled, spectres_val_scaled),
    epochs=500,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ],
    verbose=1
)

def build_mapper(theta_dim, latent_dim):
    mapper = Sequential([
        Dense(256, activation='gelu', input_shape=(theta_dim,)),
        Dense(256, activation='gelu'),
        Dense(256, activation='gelu'),
        Dense(256, activation='gelu'),
        Dense(latent_dim, activation='linear')
    ])
    mapper.compile(optimizer=Adam(1e-4), loss=MeanSquaredError())
    return mapper

mapper = build_mapper(theta.shape[1], latent_dim)

latent_train = encoder.predict(spectres_train_scaled, batch_size=128, verbose=0)
latent_val = encoder.predict(spectres_val_scaled, batch_size=128, verbose=0)

# 📌 Callbacks
history_mapper = mapper.fit(
    theta_train, latent_train,
    validation_data=(theta_val, latent_val),
    epochs=500,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ],
    verbose=1
)


# 📌 Sauvegarde du modèle
#autoencoder.save("autoencoder_model.keras")

# 📌 Plot de la loss
def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('mapper_loss.png', dpi=150)
    plt.close()

plot_loss(history_ae)


def reconstruct_from_theta(theta_input):
    latent_pred = mapper.predict(theta_input, verbose=0)
    decoded_scaled = decoder.predict(latent_pred, verbose=0)
    decoded_log = scaler_spectre.inverse_transform(decoded_scaled)
    decoded_orig = np.expm1(decoded_log)
    return decoded_orig


# 📌 Évaluation globale
def plot_global_errors_per_theta(mapper, decoder, scaler_spectre, theta_input, spectres_true_scaled, custom_legends=None):
    eps = 1e-8
    spectres_pred = reconstruct_from_theta(theta_input)
    spectres_true = np.expm1(scaler_spectre.inverse_transform(spectres_true_scaled))

    diffs = spectres_true - spectres_pred
    norm_diff = np.linalg.norm(diffs, axis=1)
    norm_true = np.linalg.norm(spectres_true, axis=1) + eps
    global_errors = 100 * norm_diff / norm_true

    mae = np.mean(np.abs(diffs))
    max_error = np.max(np.abs(diffs))

    print(f"\nNombre de spectres : {len(global_errors)}")
    print(f"Erreur globale moyenne : {np.mean(global_errors):.3f}%")
    print(f"Erreur absolue moyenne (MAE) : {mae:.4f}")
    print(f"Erreur absolue max : {max_error:.4f}")
    print(f"Spectres > 5% erreur : {np.sum(global_errors > 5)}")

    plt.figure(figsize=(5 * theta_input.shape[1], 4))
    theta_names = custom_legends or [f"$\\theta_{i}$" for i in range(theta_input.shape[1])]

    for i in range(theta_input.shape[1]):
        plt.subplot(1, theta_input.shape[1], i + 1)
        plt.scatter(theta_input[:, i], global_errors, color='dodgerblue', alpha=0.7, edgecolors='black', s=40)
        xlabel = theta_names[i] if isinstance(theta_names[i], str) else theta_names[i]["xlabel"]
        plt.xlabel(xlabel)
        plt.ylabel("Global Error (%)")

    plt.tight_layout()
    plt.savefig("mapper_error.png")
    plt.close()

    return global_errors

# 📌 Custom légendes pour theta
custom_legends = [r"$kT_s$", r"$Z$", r"$\tau$", r"$norm$"]

# 📌 Évaluation
spectres_scaled_full = np.concatenate([spectres_train_scaled, spectres_val_scaled], axis=0)
theta_full = np.concatenate([theta_train, theta_val], axis=0)
plot_global_errors_per_theta(mapper, decoder, scaler_spectre, theta_full, spectres_scaled_full, custom_legends)

def plot_sample_from_theta(idx, theta_input, spectres_scaled, scaler_spectre, mapper, decoder):
    theta_sample = theta_input[idx:idx+1]
    spectrum_pred = reconstruct_from_theta(theta_sample).flatten()
    spectrum_true = np.expm1(scaler_spectre.inverse_transform(spectres_scaled[idx:idx+1])).flatten()

    eps = 1e-8
    error_bin = (spectrum_true - spectrum_pred) / (np.abs(spectrum_true) + eps) * 100
    global_error = 100 * np.linalg.norm(spectrum_true - spectrum_pred) / np.linalg.norm(spectrum_true)

    print(f"✅ Sample {idx} | Relative global error : {global_error:.3f}%")

    e_min, e_max, num_bins = 0.1, 50, spectrum_true.shape[0]
    energy_array = np.geomspace(e_min, e_max, num_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    ax1.semilogx(energy_array, spectrum_pred, label='Reconstructed', color='blue')
    ax1.semilogx(energy_array, spectrum_true, label='Original', color='red', linestyle='dashed')
    ax1.set_xlabel("Energy (keV)")
    ax1.set_ylabel("Intensity")
    ax1.legend()
    ax1.set_title(f"Autoencoder - Sample {idx}", fontsize=10)

    ax2.plot(energy_array, error_bin, color='red', label='Relative error (%)')
    ax2.fill_between(energy_array, 0, error_bin, color='red', alpha=0.3)
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Error (%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('mapper_reconstruction.png')
    plt.close()

# 📌 Custom légendes pour theta
custom_legends = [r"$kT_s$", r"$Z$", r"$\tau$", r"$norm$"]

# 📌 Évaluation
spectres_scaled_full = np.concatenate([spectres_train_scaled, spectres_val_scaled], axis=0)
theta_full = np.concatenate([theta_train, theta_val], axis=0)
plot_global_errors_per_theta(mapper, decoder, scaler_spectre, theta_full, spectres_scaled_full, custom_legends)

# 📌 Affichage d’un exemple
plot_sample_from_theta(20, theta_full, spectres_scaled_full, scaler_spectre, mapper, decoder)



def plot_mean_error_per_bin_from_theta(mapper, decoder, scaler_spectre, theta_input, spectres_scaled):
    spectres_pred = reconstruct_from_theta(theta_input)
    spectres_true = np.expm1(scaler_spectre.inverse_transform(spectres_scaled))

    eps = 1e-8
    relative_errors = np.abs(spectres_true - spectres_pred) / (np.abs(spectres_true) + eps)
    mean_error_per_bin = 100 * np.mean(relative_errors, axis=0)

    e_min, e_max, n_bins = 0.1, 50, spectres_true.shape[1]
    energy_array = np.geomspace(e_min, e_max, n_bins)

    plt.semilogx(energy_array, mean_error_per_bin)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Mean Relative Error (%)")
    plt.xlim(0.1, 10)
    plt.autoscale(axis='y')
    plt.tight_layout()
    plt.savefig("mapper_bin.png")
    plt.close()

    return mean_error_per_bin

# 📌 Appel
spectres_scaled_full = np.concatenate([spectres_train_scaled, spectres_val_scaled], axis=0)
theta_full = np.concatenate([theta_train, theta_val], axis=0)
plot_mean_error_per_bin_from_theta(mapper, decoder, scaler_spectre, theta_full, spectres_scaled_full)




