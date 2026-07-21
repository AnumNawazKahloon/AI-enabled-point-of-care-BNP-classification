"""Train the 5-CNN ensemble, combine via voting/stacking, evaluate. Run: python train.py"""
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
import data_utils
import models as model_defs
import ensemble
import evaluate

np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)
tf.keras.mixed_precision.set_global_policy('mixed_float16')


class TestSetCallback(keras.callbacks.Callback):
    """Evaluates on the held-out TEST set at the end of every epoch."""
    def __init__(self, X_test, y_test):
        super().__init__()
        self.X_test, self.y_test = X_test, y_test
        self.test_loss, self.test_accuracy = [], []

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.test_loss.append(loss)
        self.test_accuracy.append(acc)


def train_all_models(data):
    X_tr_aug, y_tr_aug = data_utils.augment(data['X_tr_r'], data['y_tr_cat'],
                                             noise_std=0.015, max_shift=1, ratio=0.5)
    print(f'After augmentation — Train: {X_tr_aug.shape}')

    trained_models, val_accuracies, histories, test_histories = {}, {}, {}, {}
    for name, builder in model_defs.BUILDERS.items():
        print(f'\nTraining {name}')
        tf.keras.backend.clear_session()
        tf.random.set_seed(config.SEED)

        model = builder()
        save_path = os.path.join(config.MODEL_DIR, f'{name}.keras')
        test_cb = TestSetCallback(data['X_test_r'], data['y_test_cat'])

        t0 = time.time()
        hist = model.fit(
            X_tr_aug, y_tr_aug, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=2,
            validation_data=(data['X_val_r'], data['y_val_cat']),
            callbacks=model_defs.get_callbacks(save_path) + [test_cb],
            class_weight=data['cw_dict'],
        )
        best_acc = max(hist.history['val_accuracy'])
        trained_models[name] = model
        val_accuracies[name] = best_acc
        histories[name] = hist.history
        test_histories[name] = {'loss': test_cb.test_loss, 'accuracy': test_cb.test_accuracy}
        print(f'{name} — Best Val Acc: {best_acc:.4f}  ({(time.time()-t0)/60:.1f} min)')

    return trained_models, val_accuracies, histories, test_histories


def main():
    data = data_utils.prepare_data()
    trained_models, val_accuracies, histories, test_histories = train_all_models(data)

    with open(os.path.join(config.OUTPUT_DIR, 'histories.pkl'), 'wb') as f:
        pickle.dump({'train': histories, 'test': test_histories, 'val_acc': val_accuracies}, f)

    # ── Collect test-set predictions ──────────────────────────
    test_probas = {name: m.predict(data['X_test_r'], verbose=0) for name, m in trained_models.items()}
    y_test_int = np.argmax(data['y_test_cat'], axis=1)

    results = {}
    for name, proba in test_probas.items():
        results[name] = ensemble.report(name, y_test_int, np.argmax(proba, axis=1))

    soft_avg, pred_soft = ensemble.soft_voting(test_probas)
    results['Soft Voting (equal)'] = ensemble.report('Soft Voting', y_test_int, pred_soft)

    weighted_avg, pred_weighted, weights = ensemble.weighted_voting(test_probas, val_accuracies)
    results['Weighted Voting'] = ensemble.report('Weighted Voting', y_test_int, pred_weighted)

    pred_hard = ensemble.hard_voting(test_probas)
    results['Hard Voting'] = ensemble.report('Hard Voting', y_test_int, pred_hard)

    temp_avg, pred_temp = ensemble.temp_scaled_voting(test_probas, T=1.5)
    results['Temp-Scaled Voting'] = ensemble.report('Temp-Scaled Voting', y_test_int, pred_temp)

    meta_lr, pred_stack = ensemble.fit_stacking_meta_learner(test_probas, y_test_int)
    results['CESML'] = ensemble.report('Stacking (CESML)', y_test_int, pred_stack)
    import joblib
    joblib.dump(meta_lr, config.META_LEARNER_PATH)

    # ── Evaluation & plots ─────────────────────────────────────
    sorted_results = evaluate.results_comparison(results)
    best_method = sorted_results[0][0]
    best_pred = {'Soft Voting (equal)': pred_soft, 'Weighted Voting': pred_weighted,
                 'Hard Voting': pred_hard, 'Temp-Scaled Voting': pred_temp,
                 'CESML': pred_stack}.get(best_method, np.argmax(test_probas.get(best_method, soft_avg), axis=1))
    evaluate.confusion_matrix_plot(y_test_int, best_pred, best_method)
    evaluate.diversity_analysis(test_probas, y_test_int, results['Soft Voting (equal)'])
    evaluate.roc_curves(test_probas, soft_avg, y_test_int)
    evaluate.calibration_analysis(soft_avg, y_test_int)
    evaluate.bnp_correlation_analysis(soft_avg, y_test_int)
    evaluate.eis_nyquist_bode(data['X_test'], y_test_int)

    if data['has_ext']:
        print('\nEXTERNAL TEST SET RESULTS')
        evaluate.evaluate_external(trained_models, meta_lr, weights, data['X_ext_r'], data['y_ext_cat'])

    # ── Production ensemble sanity check ───────────────────────
    prod_ensemble = ensemble.CNNEnsemble(trained_models, data['scaler'], meta_lr,
                                         val_accuracies, method='weighted')
    preds, probas = prod_ensemble.predict(data['X_test'][:5])
    print(f'\nSample production predictions: {preds}')


if __name__ == '__main__':
    main()
