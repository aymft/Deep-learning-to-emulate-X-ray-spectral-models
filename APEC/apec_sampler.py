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


plt.rcParams['axes.unicode_minus'] = False


# ── 0) Liste de vos fichiers .npz et containers ───────────────────────────────
file_paths = [
    "/tmpdir/ferec/apec_log7_0.1-0.3keV.npz",
    "/tmpdir/ferec/apec_log7_0.3-0.5keV.npz",
    "/tmpdir/ferec/apec_log7_0.5-0.7keV.npz",
    "/tmpdir/ferec/apec_log7_0.7-0.9keV.npz",
    "/tmpdir/ferec/apec_log7_0.9-1.0keV.npz",
    "/tmpdir/ferec/apec_log7_1.0-3.0keV.npz",
    "/tmpdir/ferec/apec_log7_3.0-5.0keV.npz",
    "/tmpdir/ferec/apec_log7_5.0-7.0keV.npz",
    "/tmpdir/ferec/apec_log7_7.0-9.0keV.npz",
    "/tmpdir/ferec/apec_log7_9.0-11.0keV.npz",
    # "/tmpdir/ferec/apec_11.0-13.0keV.npz",
    # "/tmpdir/ferec/apec_13.0-15.0keV.npz",
    # "/tmpdir/ferec/apec_15.0-17.0keV.npz",
    # "/tmpdir/ferec/apec_17.0-19.0keV.npz",
    # "/tmpdir/ferec/apec_19.0-21.0keV.npz",
    # "/tmpdir/ferec/apec_21.0-23.0keV.npz",
    # "/tmpdir/ferec/apec_23.0-25.0keV.npz",
    # "/tmpdir/ferec/apec_25.0-27.0keV.npz",
    # "/tmpdir/ferec/apec_27.0-29.0keV.npz",
    # "/tmpdir/ferec/apec_29.0-31.0keV.npz",
    # "/tmpdir/ferec/apec_31.0-33.0keV.npz",
    # "/tmpdir/ferec/apec_33.0-35.0keV.npz",
    # "/tmpdir/ferec/apec_35.0-37.0keV.npz",
    # "/tmpdir/ferec/apec_37.0-39.0keV.npz",
    # "/tmpdir/ferec/apec_39.0-41.0keV.npz",
    # "/tmpdir/ferec/apec_41.0-43.0keV.npz",
    # "/tmpdir/ferec/apec_43.0-45.0keV.npz",
    # "/tmpdir/ferec/apec_45.0-47.0keV.npz",
    # "/tmpdir/ferec/apec_47.0-49.0keV.npz",
    # "/tmpdir/ferec/apec_49.0-50.0keV.npz",
]

models       = []
X_tests      = []
y_tests      = []
scalers_spec = []
energies = []
labels = [f"Bande {i+1}" for i in range(len(file_paths))]


# ── 1) Fonction de construction du modèle ─────────────────────────────────────
def build_surrogate_model(input_dim, output_dim, n_units=128):
    inp = Input(shape=(input_dim,))
    x = Dense(n_units, activation='gelu')(inp)
    x = Dense(n_units, activation='gelu')(x)
    x = Dense(n_units, activation='gelu')(x)
    c = Dense(n_units, activation='gelu')(x)
    cont = Dense(output_dim, activation='linear', name='continuum')(c)
    e = Dense(n_units, activation='gelu')(x)
    emis = Dense(output_dim, activation='softplus', name='emission')(e)
    out = Add()([cont, emis])
    return Model(inp, out)

# ── 2) Votre loss custom ───────────────────────────────────────────────────────
def improved_spectral_loss(y_true, y_pred):
    mse      = tf.reduce_mean(tf.square(y_true - y_pred))
    dy_true  = y_true[:,1:] - y_true[:,:-1]
    dy_pred  = y_pred[:,1:] - y_pred[:,:-1]
    grad     = tf.reduce_mean(tf.square(dy_true - dy_pred))
    ddy_true = dy_true[:,1:] - dy_true[:,:-1]
    ddy_pred = dy_pred[:,1:] - dy_pred[:,:-1]
    curv     = tf.reduce_mean(tf.square(ddy_true - ddy_pred))
    return mse + grad + 2.0 * curv

# ── 3) Boucle sur les fichiers pour charger + entraîner ────────────────────────
for path, label in zip(file_paths, labels):
    # 3.a) Chargement
    data      = np.load(path)
    specs     = data["spectra"]          # shape (N, n_bins)
    theta     = data["params"][:, :2]    # kT, Z
    energy    = data["energy"]           # same for all
    energies.append(energy)

    # 3.b) Log-transform + scaling spectres
    specs_lp  = np.log1p(specs)
    scaler    = StandardScaler().fit(specs_lp)
    specs_s   = scaler.transform(specs_lp)

    # 3.c) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        theta, specs_s, test_size=0.1, random_state=42
    )

    # 3.d) Build & compile
    model = build_surrogate_model(input_dim=theta.shape[1], output_dim=specs_s.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=improved_spectral_loss
    )

    # 3.e) Callbacks & fit
    es  = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=[es, rlr],
        verbose=1
    )

    # 3.f) Sauvegarde dans les lists
    models.append(model)
    X_tests.append(X_te)
    y_tests.append(y_te)
    scalers_spec.append(scaler)

print("✅ Entraînement des 3 modèles terminé.")



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scienceplots

def plot_concat_styled(idx):
    """
    Affiche en deux subplots (2/3 pour le spectre concaténé, 1/3 pour l'erreur concaténée),
    avec le style IEEE (science, no-latex), en semilogy pour les spectres,
    et en abscisse les vraies énergies (keV).
    """
    # 1) Préparation des données
    eps = 1e-8
    true_list, pred_list, err_list, energy_list = [], [], [], []
    boundaries = []
    offset = 0

    for label, model, X_te, y_te, scaler, energy_array in zip(
        labels, models, X_tests, y_tests, scalers_spec, energies
    ):
        th       = X_te[idx:idx+1]
        true_s   = y_te[idx]
        pred_s   = model.predict(th)[0]

        # inversion & linéaire
        true_lin = scaler.inverse_transform(true_s.reshape(1, -1))[0]
        pred_lin = scaler.inverse_transform(pred_s.reshape(1, -1))[0]

        # si vous aviez log1p :
        # true_lin = np.expm1(true_lin)
        # pred_lin = np.expm1(pred_lin)

        err_bin = (true_lin - pred_lin) / (np.abs(true_lin) + eps) * 100

        n_bins = len(true_lin)
        boundaries.append(offset)
        true_list.append(true_lin)
        pred_list.append(pred_lin)
        err_list.append(err_bin)
        energy_list.append(energy_array)
        offset += n_bins

    boundaries.append(offset)

    # vecteurs concaténés
    x_all    = np.concatenate(energy_list)   # → vraies énergies
    true_all = np.concatenate(true_list)
    pred_all = np.concatenate(pred_list)
    err_all  = np.concatenate(err_list)

    # 2) Style IEEE
    plt.style.use(['science', 'no-latex'])
    rcParams['figure.figsize']    = (5, 4)
    rcParams['figure.dpi']        = 1000
    rcParams['font.size']         = 9
    rcParams['axes.titlesize']    = 9
    rcParams['axes.labelsize']    = 9
    rcParams['xtick.labelsize']   = 7
    rcParams['ytick.labelsize']   = 7
    rcParams['legend.fontsize']   = 9
    rcParams['lines.linewidth']   = 1.0
    rcParams['grid.linewidth']    = 0.5

    # 3) Création des subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # ─── Spectres (semilogy) ─────────────────────────────────────────────
    ax1.loglog(x_all, pred_all, label='Reconstructed spectrum')
    ax1.loglog(x_all, true_all, linestyle='--', label='Original spectrum')
    ax1.set_ylabel(r"Flux $(\mathrm{erg\,cm^{-2}\,s^{-1}})$")
    ax1.legend()


    # ─── Erreur concaténée (linéaire) ───────────────────────────────────
    ax2.semilogx(x_all, err_all, label='Relative Error (%)')
    ax2.fill_between(x_all, err_all, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Error (%)")


    plt.tight_layout()
    plt.savefig("plot_concat.pdf")
    plt.show()

# Exemple d'appel
plot_concat_styled(77)



import numpy as np
import matplotlib.pyplot as plt

def plot_error_by_param_all():
    """
    Trace en une figure 1×2 :
      - à gauche : erreur globale vs kT pour chaque set
      - à droite : erreur globale vs abondance pour chaque set
    """
    plt.style.use(['science', 'no-latex', 'scatter'])

    # rcParams['figure.figsize']    = (5, 4)
    rcParams['figure.dpi']        = 1000
    rcParams['font.size']         = 9
    rcParams['axes.titlesize']    = 9
    rcParams['axes.labelsize']    = 9
    rcParams['xtick.labelsize']   = 7
    rcParams['ytick.labelsize']   = 7
    rcParams['legend.fontsize']   = 9
    rcParams['lines.linewidth']   = 1.0
    rcParams['grid.linewidth']    = 0.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Parcourt chaque jeu de test / modèle
    for label, model, X_te, y_te, scaler in zip(
        labels, models, X_tests, y_tests, scalers_spec
    ):
        # Récupère les paramètres
        kT_vals = X_te[:, 0]
        Z_vals  = X_te[:, 1]

        # Calcule l'erreur globale pour chaque échantillon
        errors = []
        for th, true_s in zip(X_te, y_te):
            pred_s = model.predict(th[None, :])[0]

            # inversion du scaler + retour en linéaire
            true_lp = scaler.inverse_transform(true_s.reshape(1, -1))[0]
            pred_lp = scaler.inverse_transform(pred_s.reshape(1, -1))[0]
            true_lin = np.expm1(true_lp)
            pred_lin = np.expm1(pred_lp)

            err_glob = 100 * np.linalg.norm(true_lin - pred_lin) / np.linalg.norm(true_lin)
            errors.append(err_glob)
        errors = np.array(errors)

        # Scatter plots
        ax1.scatter(kT_vals, errors,
                    label=label, edgecolor='k', alpha=0.7)
        ax2.scatter(Z_vals,  errors,
                    label=label, edgecolor='k', alpha=0.7)

    # Mise en forme axe gauche
    ax1.set_xlabel("kT (keV)")
    ax1.set_ylabel("Erreur globale (%)")
    ax1.grid(True, ls=':', alpha=0.5)
    ax1.legend()

    # Mise en forme axe droit
    ax2.set_xlabel("Abondance")
    ax2.grid(True, ls=':', alpha=0.5)
    # On peut omettre la légende répétée à droite si on veut

    plt.savefig("error_concat.pdf")
    plt.tight_layout()
    plt.show()

# Exemple d'appel :
plot_error_by_param_all()
