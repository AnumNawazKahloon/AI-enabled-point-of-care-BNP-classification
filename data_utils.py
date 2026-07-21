"""Data loading, patient-based train/test/val split, scaling, augmentation."""
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight

import config


def load_csv(path):
    """Load a headerless CSV, coerce to numeric, drop NaN rows. Last column = label."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Cannot find {path}')
    df = pd.read_csv(path, header=None, low_memory=False)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    print(f'Loaded {path}: {df.shape}')
    return df.iloc[:, :-1].values, df.iloc[:, -1].values.astype(int)


def load_patient_ids(path, n_rows):
    """Load patient IDs from a separate CSV, aligned by row order to the feature file."""
    ids = pd.read_csv(path, header=None).values.ravel()
    if len(ids) != n_rows:
        raise ValueError(f'patient_ids ({len(ids)}) and feature rows ({n_rows}) mismatch')
    return ids


def patient_based_split(X, y, patient_ids, train_frac, test_frac, val_frac,
                         min_patients_per_class=3, seed=42):
    """Split by unique patient so no patient's rows appear in more than one split.
    Each patient is assigned to a class via majority vote of its own labels, then
    patients (not rows) are divided per-class into train/test/val fractions.
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)

    patient_label = {}
    for pid in unique_patients:
        labels = y[patient_ids == pid]
        patient_label[pid] = np.bincount(labels).argmax()

    train_p, test_p, val_p, dropped = [], [], [], {}
    for cls in np.unique(y):
        cls_patients = np.array([p for p in unique_patients if patient_label[p] == cls])
        n = len(cls_patients)
        if n < min_patients_per_class:
            dropped[int(cls)] = n
            continue
        cls_patients = cls_patients.copy()
        rng.shuffle(cls_patients)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_train = n - n_test - n_val
        train_p.extend(cls_patients[:n_train])
        test_p.extend(cls_patients[n_train:n_train + n_test])
        val_p.extend(cls_patients[n_train + n_test:])

    train_p, test_p, val_p = set(train_p), set(test_p), set(val_p)
    train_idx = np.array([i for i, p in enumerate(patient_ids) if p in train_p])
    test_idx = np.array([i for i, p in enumerate(patient_ids) if p in test_p])
    val_idx = np.array([i for i, p in enumerate(patient_ids) if p in val_p])
    rng.shuffle(train_idx); rng.shuffle(test_idx); rng.shuffle(val_idx)

    return (X[train_idx], X[test_idx], X[val_idx],
            y[train_idx], y[test_idx], y[val_idx], dropped)


def augment(X, y_cat, noise_std=0.015, max_shift=1, ratio=0.5):
    """Add Gaussian noise + random time-axis shift to a fraction of the data."""
    n = int(len(X) * ratio)
    idx = np.random.choice(len(X), n, replace=False)
    Xa = X[idx].copy()
    ya = y_cat[idx].copy()

    Xa += np.random.normal(0, noise_std, Xa.shape)

    t_len = X.shape[1]
    for i in range(len(Xa)):
        s = np.random.randint(-max_shift, max_shift + 1)
        if s > 0:
            Xa[i, s:, :, :] = Xa[i, :t_len - s, :, :]
            Xa[i, :s, :, :] = 0
        elif s < 0:
            Xa[i, :t_len + s, :, :] = Xa[i, -s:, :, :]
            Xa[i, t_len + s:, :, :] = 0

    Xout = np.concatenate([X, Xa], axis=0)
    yout = np.concatenate([y_cat, ya], axis=0)
    perm = np.random.permutation(len(Xout))
    return Xout[perm], yout[perm]


def prepare_data():
    """Load, patient-split, scale, reshape, one-hot encode, compute class weights."""
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    X_raw, y_raw = load_csv(config.TRAIN_FILE)
    patient_ids = load_patient_ids(config.PATIENT_ID_FILE, len(X_raw))
    print(f'Features: {X_raw.shape[1]}, Classes: {np.unique(y_raw)}, '
          f'Unique patients: {len(np.unique(patient_ids))}')

    X_tr, X_test, X_val, y_tr, y_test, y_val, dropped = patient_based_split(
        X_raw, y_raw, patient_ids,
        config.TRAIN_FRAC, config.TEST_FRAC, config.VAL_FRAC,
        min_patients_per_class=config.MIN_PATIENTS_PER_CLASS, seed=config.SEED
    )
    if dropped:
        print(f'Dropped classes with < {config.MIN_PATIENTS_PER_CLASS} patients: {dropped}')

    n_total = len(X_raw)
    print(f'Train: {len(X_tr)} ({len(X_tr)/n_total:.1%})  |  '
          f'Test: {len(X_test)} ({len(X_test)/n_total:.1%})  |  '
          f'Val: {len(X_val)} ({len(X_val)/n_total:.1%})')

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_test_s = scaler.transform(X_test)
    X_val_s = scaler.transform(X_val)
    joblib.dump(scaler, config.SCALER_PATH)

    X_tr_r = X_tr_s.reshape(-1, *config.INPUT_SHAPE)
    X_test_r = X_test_s.reshape(-1, *config.INPUT_SHAPE)
    X_val_r = X_val_s.reshape(-1, *config.INPUT_SHAPE)

    y_tr_cat = keras.utils.to_categorical(y_tr, config.N_CLASSES)
    y_test_cat = keras.utils.to_categorical(y_test, config.N_CLASSES)
    y_val_cat = keras.utils.to_categorical(y_val, config.N_CLASSES)

    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = dict(enumerate(cw))

    data = dict(
        X_tr=X_tr, X_test=X_test, X_val=X_val,
        X_tr_r=X_tr_r, X_test_r=X_test_r, X_val_r=X_val_r,
        y_tr=y_tr, y_test=y_test, y_val=y_val,
        y_tr_cat=y_tr_cat, y_test_cat=y_test_cat, y_val_cat=y_val_cat,
        scaler=scaler, cw_dict=cw_dict,
    )

    # Optional external test set (already held out — no split needed)
    try:
        X_ext, y_ext = load_csv(config.EXTERNAL_FILE)
        X_ext_s = scaler.transform(X_ext)
        data['X_ext_r'] = X_ext_s.reshape(-1, *config.INPUT_SHAPE)
        data['y_ext_cat'] = keras.utils.to_categorical(y_ext, config.N_CLASSES)
        data['has_ext'] = True
        print(f'External test: {data["X_ext_r"].shape}')
    except FileNotFoundError:
        data['has_ext'] = False
        print('No external test set found.')

    return data
