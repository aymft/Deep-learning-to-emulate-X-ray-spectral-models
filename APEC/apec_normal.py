import numpy as np
import tensorflow as tf

from tensorflow.keras.models     import Sequential, Model
from tensorflow.keras.layers     import Input, Dense, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing       import StandardScaler
from sklearn.model_selection     import train_test_split
from pathlib                     import Path
import matplotlib.pyplot         as plt

from matplotlib import rcParams
import scienceplots
import time

# juste après vos rcParams, ou tout en haut du script :
plt.rcParams['axes.unicode_minus'] = False


t0 = time.time()
# ── 1) Chargement des données ─────────────────────────────────────────────────
data_path    = Path("/tmpdir/ferec/apec_log1e5_7.0-9.0keV.npz")
data         = np.load(data_path)
spectres     = data["spectra"]          # intensités brutes
theta        = data["params"][:, :2]    # kT et abondance seulement
energy_array = data["energy"]

# ── 2) Log-transform des spectres ────────────────────────────────────────────
spectres_lp = spectres
# spectres_lp = np.log1p(spectres)

# ── 3) Scalers (fit sur tout le jeu) ──────────────────────────────────────────
#scaler_theta  = StandardScaler().fit(theta)
scaler_spec   = StandardScaler().fit(spectres_lp)

theta_scaled    = theta
#scaler_theta.transform(theta)
spectres_scaled = scaler_spec.transform(spectres_lp)

# ── 4) Split train/test ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    theta_scaled,
    spectres_scaled,
    test_size=0.1,
    random_state=42
)

# ── 5) Définition du modèle ───────────────────────────────────────────────────
def build_surrogate_model(input_dim, output_dim):
    inputs  = Input(shape=(input_dim,), name="input_layer")
    n_units = 128

    x = Dense(n_units, activation='gelu')(inputs)
    x = Dense(n_units, activation='gelu')(x)
    x = Dense(n_units, activation='gelu')(x)

    # Continuum
    c = Dense(n_units, activation='gelu')(x)
    cont = Dense(output_dim, activation='linear', name='continuum')(c)

    # Emission
    e = Dense(n_units, activation='gelu')(x)
    emis = Dense(output_dim, activation='softplus', name='emission')(e)

    total = Add(name='total_spectrum')([cont, emis])
    return Model(inputs, total, name="SurrogateAPEC")

input_dim  = X_train.shape[1]
output_dim = y_train.shape[1]
model      = build_surrogate_model(input_dim, output_dim)

# ── 6) Compilation avec loss custom ────────────────────────────────────────────
def improved_spectral_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    dy_true = y_true[:,1:] - y_true[:,:-1]
    dy_pred = y_pred[:,1:] - y_pred[:,:-1]
    grad = tf.reduce_mean(tf.square(dy_true - dy_pred))
    ddy_true = dy_true[:,1:] - dy_true[:,:-1]
    ddy_pred = dy_pred[:,1:] - dy_pred[:,:-1]
    curv = tf.reduce_mean(tf.square(ddy_true - ddy_pred))
    return mse + grad + 2.0 * curv


# just avant de compiler le modèle, récupérez les paramètres du scaler
mean_spec  = tf.constant(scaler_spec.mean_,  dtype=tf.float32)  # shape (n_bins,)
scale_spec = tf.constant(scaler_spec.scale_, dtype=tf.float32)  # shape (n_bins,)

def masked_spectral_loss(y_true, y_pred, eps=1e-8):
    # 1) reconstruire y_true dans son espace original : y_true_lin = y_true*scale + mean
    y_true_lin = y_true * scale_spec + mean_spec

    # 2) seuil par échantillon
    max_vals  = tf.reduce_max(y_true_lin, axis=1, keepdims=True)   # (batch,1)
    threshold = max_vals / 5e5

    # 3) masque des bins significatifs
    mask = tf.cast(y_true_lin >= threshold, tf.float32)            # (batch, n_bins)

    # 4) MSE masquée
    se  = tf.square(y_true - y_pred) * mask
    mse = tf.reduce_sum(se, axis=1) / (tf.reduce_sum(mask, axis=1) + eps)

    # 5) gradient masqué
    dy_true  = y_true[:,1:]   - y_true[:,:-1]
    dy_pred  = y_pred[:,1:]   - y_pred[:,:-1]
    mask_g   = mask[:,1:] * mask[:,:-1]
    se_g     = tf.square(dy_true - dy_pred) * mask_g
    grad     = tf.reduce_sum(se_g, axis=1) / (tf.reduce_sum(mask_g, axis=1) + eps)

    # 6) courbure masquée
    ddy_true  = dy_true[:,1:] - dy_true[:,:-1]
    ddy_pred  = dy_pred[:,1:] - dy_pred[:,:-1]
    mask_c    = mask[:,2:] * mask[:,1:-1] * mask[:,:-2]
    se_c      = tf.square(ddy_true - ddy_pred) * mask_c
    curv      = tf.reduce_sum(se_c, axis=1) / (tf.reduce_sum(mask_c, axis=1) + eps)

    # 7) combine
    loss_per_sample = mse + grad + 2.0 * curv
    return tf.reduce_mean(loss_per_sample)


model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=improved_spectral_loss
    #masked_spectral_loss
    #MeanSquaredError()
    #
)

model.summary()

# ── 7) Callbacks & entraînement ─────────────────────────────────────────────
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10,
    restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5,
    patience=5, min_lr=1e-6, verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

t1 = time.time()
print(f"⏱️ Compilation du modèle terminée en {t1-t0:.3f} secondes")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scienceplots

def plot_loss(history):
    # Appliquer le style scientifique sans LaTeX
    plt.style.use(['science', 'no-latex'])
    
    # Paramètres RC pour un rendu IEEE
    rcParams['figure.figsize'] = (3.5, 2.5)  # Taille du plot en pouces
    rcParams['figure.dpi'] = 300             # Haute résolution
    rcParams['font.size'] = 8                # Taille des polices
    rcParams['axes.titlesize'] = 8           # Taille du titre des axes
    rcParams['axes.labelsize'] = 8           # Taille des labels des axes
    rcParams['xtick.labelsize'] = 7          # Taille des ticks sur l'axe X
    rcParams['ytick.labelsize'] = 7          # Taille des ticks sur l'axe Y
    rcParams['legend.fontsize'] = 7          # Taille de la légende
    rcParams['lines.linewidth'] = 1.0        # Épaisseur des courbes
    rcParams['grid.linewidth'] = 0.5         # Épaisseur des lignes de la grille
    
    # Création du graphique
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()  # Ajustement automatique des marges
    plt.savefig("apec_loss.png")
   

plot_loss(history)

def plot_sample_styled(idx):
    # 1) Récupération des données scalées et prédiction
    th_scaled   = X_test[idx:idx+1]
    true_scaled = y_test[idx]
    pred_scaled = model.predict(th_scaled)[0]

    # 2) Inversion du scaler
    true_lin = scaler_spec.inverse_transform(true_scaled.reshape(1, -1))[0]
    pred_lin = scaler_spec.inverse_transform(pred_scaled.reshape(1, -1))[0]

    # 3) Calcul de l'erreur
    eps      = 1e-8
    err_bin  = (true_lin - pred_lin) / (np.abs(true_lin) + eps) * 100
    err_glob = 100 * np.linalg.norm(true_lin - pred_lin) / (np.linalg.norm(true_lin) + eps)
    print(f"Sample {idx} | Relative overall error: {err_glob:.3f}%\n")

    # 4) Style IEEE
    plt.style.use(['science', 'no-latex'])
    rcParams['figure.figsize']    = (5, 4)
    rcParams['figure.dpi']        = 300
    rcParams['font.size']         = 8
    rcParams['axes.titlesize']    = 8
    rcParams['axes.labelsize']    = 8
    rcParams['xtick.labelsize']   = 7
    rcParams['ytick.labelsize']   = 7
    rcParams['legend.fontsize']   = 7
    rcParams['lines.linewidth']   = 1.0
    rcParams['grid.linewidth']    = 0.5

    # 5) Création des subplots (2/3 pour le spectre, 1/3 pour l'erreur)
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # ─── Spectres (semilogx) ────────────────────────────────────────────────
    ax1.semilogy(energy_array, pred_lin, label='Reconstructed Spectrum')
    ax1.semilogy(energy_array, true_lin, linestyle='--', label='Original Spectrum')
    ax1.set_ylabel(r"Flux $(\mathrm{erg\,cm^{-2}\,s^{-1}})$")
    ax1.legend()


    # ─── Erreur bin par bin (semilogx) ────────────────────────────────────
    ax2.plot(energy_array, err_bin, label='Relative Error (%)')
    ax2.fill_between(energy_array, err_bin, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Error (%)")
    
    plt.savefig("spectra_apec.png")
    plt.tight_layout()
    plt.show()


# Exemple d'appel
plot_sample_styled(77)



def plot_error_by_param_styled():
    """
    Affiche en deux subplots (IEEE style) :
      - à gauche : erreur globale vs kT
      - à droite : erreur globale vs abondance
    et indique l'indice du spectre avec l'erreur maximale.
    """
    # 1) Calcul des erreurs globales
    errors = []
    for th, true_s in zip(X_test, y_test):
        pred_s    = model.predict(th[None, :])[0]
        true_lin  = scaler_spec.inverse_transform(true_s.reshape(1, -1))[0]
        pred_lin  = scaler_spec.inverse_transform(pred_s.reshape(1, -1))[0]
        err_glob  = 100 * np.linalg.norm(true_lin - pred_lin) / (np.linalg.norm(true_lin) + 1e-8)
        errors.append(err_glob)
    errors = np.array(errors)

    # 2) Indice et valeur de l'erreur max
    idx_max = np.argmax(errors)
    max_err = errors[idx_max]
    print(f"Spectre avec l'erreur maximale : idx = {idx_max}, erreur = {max_err:.2f}%")

    # 3) Paramètres
    kT_vals = X_test[:, 0]
    Z_vals  = X_test[:, 1]

    # 4) Style IEEE
    plt.style.use(['science', 'no-latex', 'scatter'])
    rcParams['figure.dpi']        = 300
    rcParams['font.size']         = 10
    rcParams['axes.titlesize']    = 10
    rcParams['axes.labelsize']    = 10
    rcParams['xtick.labelsize']   = 10
    rcParams['ytick.labelsize']   = 10
    rcParams['legend.fontsize']   = 7
    rcParams['lines.linewidth']   = 1.0
    rcParams['grid.linewidth']    = 0.5

    # 5) Création des subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    # — Erreur vs kT
    ax1.scatter(kT_vals, errors, color='dodgerblue', edgecolor='k', alpha=0.8 ,s=30, linewidths=0.8)
    ax1.set_xlabel("kT (keV)")
    ax1.set_ylabel("Global error (%)")


    # — Erreur vs Abondance
    ax2.scatter(Z_vals, errors, color='dodgerblue', edgecolor='k', alpha=0.8 ,s=30, linewidths=0.8)
    ax2.set_xlabel("Abundances")

    plt.savefig("apec_errors.png")
    plt.tight_layout()
    plt.show()

# Exemple d'appel
plot_error_by_param_styled()

