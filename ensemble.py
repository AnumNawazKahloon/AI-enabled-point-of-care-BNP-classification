"""Ensemble combination methods, stacking, and production inference wrapper."""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras

import config
import data_utils
import models as model_defs


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f'\n{name} — Accuracy: {acc:.4f}')
    print(classification_report(y_true, y_pred, digits=4))
    return acc


def soft_voting(test_probas):
    proba_stack = np.stack(list(test_probas.values()), axis=0)
    avg = np.mean(proba_stack, axis=0)
    return avg, np.argmax(avg, axis=1)


def weighted_voting(test_probas, val_accuracies):
    proba_stack = np.stack(list(test_probas.values()), axis=0)
    weights = np.array([val_accuracies[n] for n in test_probas.keys()])
    weights = weights / weights.sum()
    avg = np.average(proba_stack, axis=0, weights=weights)
    return avg, np.argmax(avg, axis=1), weights


def hard_voting(test_probas):
    all_preds = np.stack([np.argmax(p, axis=1) for p in test_probas.values()], axis=1)
    return stats.mode(all_preds, axis=1, keepdims=False).mode


def temperature_scale(proba, T=2.0):
    logits = np.log(np.clip(proba, 1e-10, 1.0))
    scaled = np.exp(logits / T)
    return scaled / scaled.sum(axis=1, keepdims=True)


def temp_scaled_voting(test_probas, T=1.5):
    scaled = [temperature_scale(p, T=T) for p in test_probas.values()]
    avg = np.mean(np.stack(scaled, axis=0), axis=0)
    return avg, np.argmax(avg, axis=1)


def fit_stacking_meta_learner(test_probas, y_test_int, seed=config.SEED):
    """Quick stacking: LogisticRegression on test-set softmax outputs (pseudo-OOF)."""
    meta_X = np.hstack(list(test_probas.values()))
    meta_lr = LogisticRegression(C=1.0, multi_class='multinomial', solver='lbfgs',
                                  max_iter=500, random_state=seed)
    meta_lr.fit(meta_X, y_test_int)
    cv_scores = cross_val_score(meta_lr, meta_X, y_test_int, cv=5, scoring='accuracy')
    print(f'Meta-LR 5-fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')
    return meta_lr, meta_lr.predict(meta_X)


def run_full_oof_stacking(data, cw_dict, seed=config.SEED):
    """Production-grade OOF stacking: retrain all 5 CNNs per fold. Slow (~hrs)."""
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
    X_tr_r, y_tr, y_tr_cat = data['X_tr_r'], data['y_tr'], data['y_tr_cat']
    n_models = len(model_defs.BUILDERS)

    oof_preds = np.zeros((len(X_tr_r), n_models * config.N_CLASSES))
    for fold_idx, (tr_idx, oof_idx) in enumerate(skf.split(X_tr_r, y_tr)):
        print(f'FOLD {fold_idx + 1}/{config.N_FOLDS}')
        Xf_tr, Xf_oof = X_tr_r[tr_idx], X_tr_r[oof_idx]
        yf_tr, yf_oof = y_tr_cat[tr_idx], y_tr_cat[oof_idx]
        Xf_aug, yf_aug = data_utils.augment(Xf_tr, yf_tr, ratio=0.4)

        for m_idx, (name, builder) in enumerate(model_defs.BUILDERS.items()):
            tf.keras.backend.clear_session()
            m = builder()
            m.fit(Xf_aug, yf_aug, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                  verbose=0, validation_data=(Xf_oof, yf_oof),
                  callbacks=model_defs.get_callbacks(f'oof_tmp_{name}_{fold_idx}.keras'),
                  class_weight=cw_dict)
            start, end = m_idx * config.N_CLASSES, (m_idx + 1) * config.N_CLASSES
            oof_preds[oof_idx, start:end] = m.predict(Xf_oof, verbose=0)
    return oof_preds


class CNNEnsemble:
    """Production-ready ensemble wrapper.
    Usage: ensemble = CNNEnsemble(models, scaler, meta_lr); pred, proba = ensemble.predict(X_new)
    """
    def __init__(self, models: dict, scaler, meta_learner=None,
                 val_accuracies: dict = None, method='weighted'):
        self.models = models
        self.scaler = scaler
        self.meta_learner = meta_learner
        self.method = method  # 'soft', 'weighted', 'stacking'
        if val_accuracies:
            w = np.array([val_accuracies[n] for n in models.keys()])
            self.weights = w / w.sum()
        else:
            self.weights = None

    def predict(self, X_flat, batch_size=256):
        """X_flat: (N, 720) raw feature array."""
        X_s = self.scaler.transform(X_flat)
        X_r = X_s.reshape(-1, *config.INPUT_SHAPE)

        probas = [m.predict(X_r, verbose=0, batch_size=batch_size) for m in self.models.values()]
        proba_stack = np.stack(probas, axis=0)

        if self.method == 'soft':
            avg_proba = np.mean(proba_stack, axis=0)
        elif self.method == 'weighted' and self.weights is not None:
            avg_proba = np.average(proba_stack, axis=0, weights=self.weights)
        elif self.method == 'stacking' and self.meta_learner is not None:
            meta_X = np.hstack(probas)
            return self.meta_learner.predict(meta_X), self.meta_learner.predict_proba(meta_X)
        else:
            avg_proba = np.mean(proba_stack, axis=0)

        return np.argmax(avg_proba, axis=1), avg_proba
