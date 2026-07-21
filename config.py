"""Paths, constants, and hyperparameters."""
import os

# ── Data paths ────────────────────────────────────────────────
DATA_DIR = "users/tang-lab/Anum/BNP/"
TRAIN_FILE = os.path.join(DATA_DIR, "patient_data_5time.csv")
PATIENT_ID_FILE = os.path.join(DATA_DIR, "patient_ids.csv")
EXTERNAL_FILE = os.path.join(DATA_DIR, "patient_data.csv")  # optional external test set

# ── Output paths ──────────────────────────────────────────────
OUTPUT_DIR = "outputs/"
MODEL_DIR = os.path.join(OUTPUT_DIR, "ensemble_models/")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures/")
SCALER_PATH = os.path.join(MODEL_DIR, "ensemble_scaler.pkl")
META_LEARNER_PATH = os.path.join(MODEL_DIR, "meta_learner.pkl")

# ── Reproducibility ───────────────────────────────────────────
SEED = 42

# ── Model / data config ───────────────────────────────────────
N_CLASSES = 4
INPUT_SHAPE = (36, 10, 2)  # 36 freq points x 10 electrodes x 2 channels (Zre, Zim) = 720 features
EPOCHS = 300
BATCH_SIZE = 64
N_FOLDS = 5  # for OOF stacking

# ── Patient-based split ───────────────────────────────────────
TRAIN_FRAC, TEST_FRAC, VAL_FRAC = 0.70, 0.20, 0.10
MIN_PATIENTS_PER_CLASS = 3  # need >=1 patient in each split to keep a class

CLASS_NAMES = ['Normal\n(<100 pg/mL)', 'Mild\n(100-300 pg/mL)',
               'Moderate\n(300-900 pg/mL)', 'High\n(>900 pg/mL)']
CLASS_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
BNP_MIDPOINTS = [50, 200, 600, 1200]  # pg/mL representative value per class
