import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
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
spectres_train, spectres_val = train_test_split(spectres, test_size=0.1, random_state=42, shuffle=True)

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
        loss=MeanSquaredError(),
    )

    return autoencoder, encoder, decoder

input_dim = spectres_train_scaled.shape[1]
latent_dim = 20
autoencoder, encoder, decoder = build_autoencoder(input_dim, latent_dim)
autoencoder.summary()

# 📌 Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

# 📌 Entraînement
history = autoencoder.fit(
    spectres_train_scaled, spectres_train_scaled,
    validation_data=(spectres_val_scaled, spectres_val_scaled),
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
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
    plt.savefig('autoencoder_loss.png', dpi=150)
    plt.close()

plot_loss(history)

# 📌 Évaluation globale
def plot_global_errors_per_theta_autoencoder(autoencoder, spectres_scaled, theta, scaler_spectre, custom_legends=None):
    eps = 1e-8
    spectres_reconstructed_scaled = autoencoder.predict(spectres_scaled, batch_size=512, verbose=0)

    spectres_orig = np.expm1(scaler_spectre.inverse_transform(spectres_scaled))
    spectres_reconstructed = np.expm1(scaler_spectre.inverse_transform(spectres_reconstructed_scaled))

    diffs = spectres_orig - spectres_reconstructed
    norm_diff = np.linalg.norm(diffs, axis=1)
    norm_true = np.linalg.norm(spectres_orig, axis=1) + eps
    global_errors = 100 * norm_diff / norm_true

    # 📊 Métriques globales
    mae = np.mean(np.abs(diffs))
    max_error = np.max(np.abs(diffs))

    print(f"\nNombre de spectres : {len(global_errors)}")
    print(f"Erreur globale moyenne : {np.mean(global_errors):.3f}%")
    print(f"Erreur absolue moyenne (MAE) : {mae:.4f}")
    print(f"Erreur absolue max : {max_error:.4f}")
    print(f"Spectres > 5% erreur : {np.sum(global_errors > 5)}")

    # 📈 Erreurs vs theta
    plt.figure(figsize=(5 * theta.shape[1], 4))
    theta_names = custom_legends or [f"$\\theta_{i}$" for i in range(theta.shape[1])]

    for i in range(theta.shape[1]):
        plt.subplot(1, theta.shape[1], i + 1)
        plt.scatter(theta[:, i], global_errors, color='dodgerblue', alpha=0.7, edgecolors='black', s=40)
        xlabel = theta_names[i] if isinstance(theta_names[i], str) else theta_names[i]["xlabel"]
        plt.xlabel(xlabel)
        plt.ylabel("Global Error (%)")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('autoencoder_error.png')
    plt.close()

    return global_errors

# 📌 Custom légende pour theta
custom_legends = [r"$kT_s$"]  # tu peux étendre avec plus de paramètres si tu veux

# 📌 Evaluation sur l'ensemble d'entraînement + validation
spectres_scaled_full = np.concatenate([spectres_train_scaled, spectres_val_scaled], axis=0)
theta_full = np.concatenate([theta[train_test_split(np.arange(len(spectres)), test_size=0.1, random_state=42)[0]],
                             theta[train_test_split(np.arange(len(spectres)), test_size=0.1, random_state=42)[1]]])

global_errors = plot_global_errors_per_theta_autoencoder(
    autoencoder=autoencoder,
    spectres_scaled=spectres_scaled_full,
    theta=theta_full,
    scaler_spectre=scaler_spectre,
    custom_legends=custom_legends
)
