"""Confusion matrix, ROC, calibration, diversity, EIS/impedance, and residual plots."""
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_curve, auc, mean_absolute_error, mean_squared_error)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr, spearmanr

import config

os.makedirs(config.FIGURE_DIR, exist_ok=True)


def results_comparison(results, baseline_acc=0.7171, save_path=None):
    results = dict(results)
    results['Baseline 2D-CNN'] = baseline_acc
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f'{"Method":<30} {"Accuracy":>10} {"Delta Baseline":>15}')
    for method, acc in sorted_results:
        delta = acc - baseline_acc
        marker = ' <- BEST' if method == sorted_results[0][0] else ''
        print(f'{method:<30} {acc:>10.4f} {delta:>+15.4f}{marker}')

    fig, ax = plt.subplots(figsize=(12, 6))
    methods = [m for m, _ in sorted_results]
    accs = [a for _, a in sorted_results]
    bar_colors = ['#4CAF50' if m == sorted_results[0][0] else
                  '#FF9800' if m == 'Baseline 2D-CNN' else '#2196F3' for m in methods]
    bars = ax.barh(methods, accs, color=bar_colors, edgecolor='white', height=0.6)
    ax.axvline(x=baseline_acc, color='red', linestyle='--', linewidth=2, label='Baseline')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{acc:.4f}', va='center', fontsize=9)
    ax.set_xlabel('Accuracy'); ax.set_title('Ensemble Methods vs Baseline', fontweight='bold')
    ax.legend(); ax.set_xlim([min(accs) - 0.02, max(accs) + 0.05]); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path or os.path.join(config.FIGURE_DIR, 'comparison_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    return sorted_results


def confusion_matrix_plot(y_true, y_pred, title, save_path=None):
    print(classification_report(y_true, y_pred, digits=4))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_true, y_pred)
    labels = [f'Class {i}' for i in range(config.N_CLASSES)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels)
    axes[0].set_title(f'Confusion Matrix — {title}', fontweight='bold')
    axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Greens', ax=axes[1],
                xticklabels=labels, yticklabels=labels)
    axes[1].set_title('Normalized (Recall per Class)', fontweight='bold')
    axes[1].set_ylabel('True Label'); axes[1].set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path or os.path.join(config.FIGURE_DIR, 'confusion_matrix.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


def diversity_analysis(test_probas, y_test_int, ensemble_acc, save_path=None):
    model_names = list(test_probas.keys())
    preds_int = {n: np.argmax(p, axis=1) for n, p in test_probas.items()}
    disagree = np.zeros((len(model_names), len(model_names)))
    for i, n1 in enumerate(model_names):
        for j, n2 in enumerate(model_names):
            disagree[i, j] = np.mean(preds_int[n1] != preds_int[n2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(disagree, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=model_names, yticklabels=model_names, ax=axes[0])
    axes[0].set_title('Pairwise Disagreement Rate', fontweight='bold')

    accs = {n: accuracy_score(y_test_int, preds_int[n]) for n in model_names}
    axes[1].bar(accs.keys(), accs.values(), color=config.CLASS_COLORS * 2, edgecolor='white')
    axes[1].axhline(y=ensemble_acc, color='red', linestyle='--', label='Ensemble')
    axes[1].set_title('Individual Accuracy vs Ensemble', fontweight='bold')
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path or os.path.join(config.FIGURE_DIR, 'diversity_analysis.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


def roc_curves(test_probas, ensemble_proba, y_test_int, save_path=None):
    y_true_bin = label_binarize(y_test_int, classes=list(range(config.N_CLASSES)))
    all_probas = {'Ensemble (Soft Vote)': ensemble_proba, **test_probas}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    macro_auc = None
    for idx, (name, proba) in enumerate(all_probas.items()):
        ax = axes[idx]
        aucs = []
        for i, (c, cname) in enumerate(zip(config.CLASS_COLORS, config.CLASS_NAMES)):
            fp, tp, _ = roc_curve(y_true_bin[:, i], proba[:, i])
            a = auc(fp, tp)
            aucs.append(a)
            ax.plot(fp, tp, color=c, lw=2, label=f'{cname.splitlines()[0]} (AUC={a:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(f'{name} — ROC', fontweight='bold')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        if idx == 0:
            macro_auc = np.mean(aucs)
    plt.tight_layout()
    plt.savefig(save_path or os.path.join(config.FIGURE_DIR, 'roc_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Ensemble macro AUC (approx): {macro_auc:.4f}')


def calibration_analysis(ensemble_proba, y_test_int, save_path=None):
    y_true_bin = label_binarize(y_test_int, classes=list(range(config.N_CLASSES)))
    fig, axes = plt.subplots(1, config.N_CLASSES, figsize=(5 * config.N_CLASSES, 5))
    for i, (ax, c) in enumerate(zip(axes, config.CLASS_COLORS)):
        prob_true, prob_pred = calibration_curve(y_true_bin[:, i], ensemble_proba[:, i],
                                                  n_bins=10, strategy='quantile')
        ece = np.mean(np.abs(prob_true - prob_pred))
        ax.plot(prob_pred, prob_true, 's-', color=c, lw=2, label=f'ECE={ece:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect')
        ax.set_title(f'Class {i}', fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path or os.path.join(config.FIGURE_DIR, 'calibration_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_external(trained_models, meta_lr, weights, X_ext_r, y_ext_cat):
    y_ext_int = np.argmax(y_ext_cat, axis=1)
    ext_probas = {name: m.predict(X_ext_r, verbose=0) for name, m in trained_models.items()}
    ext_stack = np.stack(list(ext_probas.values()), axis=0)

    ext_soft_pred = np.argmax(np.mean(ext_stack, axis=0), axis=1)
    ext_weighted_pred = np.argmax(np.average(ext_stack, axis=0, weights=weights), axis=1)
    meta_X_ext = np.hstack(list(ext_probas.values()))
    ext_meta_pred = meta_lr.predict(meta_X_ext)

    ext_results = {}
    for name, pred in [('Soft Voting', ext_soft_pred), ('Weighted Voting', ext_weighted_pred),
                        ('CESML', ext_meta_pred)]:
        ext_results[name] = accuracy_score(y_ext_int, pred)
    for name, proba in ext_probas.items():
        ext_results[name] = accuracy_score(y_ext_int, np.argmax(proba, axis=1))

    for name, acc in sorted(ext_results.items(), key=lambda x: -x[1]):
        print(f'  {name:<22}: {acc:.4f}')
    return ext_results


def bnp_correlation_analysis(ensemble_proba, y_test_int, save_path=None):
    """Predicted vs measured BNP concentration (class midpoint mapping)."""
    midpoints = np.array(config.BNP_MIDPOINTS)
    bnp_pred = (ensemble_proba * midpoints).sum(axis=1)
    bnp_true = midpoints[y_test_int]

    pearson_r, _ = pearsonr(bnp_true, bnp_pred)
    spearman_r, _ = spearmanr(bnp_true, bnp_pred)
    mae = mean_absolute_error(bnp_true, bnp_pred)
    rmse = np.sqrt(mean_squared_error(bnp_true, bnp_pred))
    print(f'Pearson r={pearson_r:.3f}  Spearman r={spearman_r:.3f}  MAE={mae:.1f}  RMSE={rmse:.1f}')

    residuals = bnp_pred - bnp_true
    bias, sd = residuals.mean(), residuals.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    jitter = np.random.RandomState(config.SEED).normal(0, 15, size=len(bnp_true))
    for cls, (c, lbl) in enumerate(zip(config.CLASS_COLORS, config.CLASS_NAMES)):
        mask = y_test_int == cls
        axes[0].scatter(bnp_true[mask] + jitter[mask], bnp_pred[mask], color=c, alpha=0.65,
                        s=40, edgecolors='white', lw=0.4, label=lbl.splitlines()[0])
    axes[0].plot([0, 1400], [0, 1400], 'k--', lw=1.8, alpha=0.5, label='Ideal (y=x)')
    axes[0].set_xlabel('True BNP (pg/mL)'); axes[0].set_ylabel('Predicted BNP (pg/mL)')
    axes[0].set_title(f'Predicted vs Measured BNP\nr={pearson_r:.3f}, MAE={mae:.1f}', fontweight='bold')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].scatter(bnp_true + jitter, residuals, color='steelblue', alpha=0.5, s=30)
    axes[1].axhline(bias, color='blue', linestyle='--', label=f'Bias={bias:+.1f}')
    axes[1].axhline(bias + 1.96 * sd, color='red', linestyle=':', label='+1.96 SD')
    axes[1].axhline(bias - 1.96 * sd, color='red', linestyle=':', label='-1.96 SD')
    axes[1].set_xlabel('True BNP (pg/mL)'); axes[1].set_ylabel('Prediction Error (pg/mL)')
    axes[1].set_title('Residual Spread', fontweight='bold')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path or os.path.join(config.FIGURE_DIR, 'bnp_correlation.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    return dict(pearson_r=pearson_r, spearman_r=spearman_r, mae=mae, rmse=rmse, bias=bias)


def eis_nyquist_bode(X_test, y_test_int, save_path=None):
    """Nyquist plot from raw (unscaled) impedance features: (N, 720) -> (N, 36, 10, 2)."""
    n_freq = config.INPUT_SHAPE[0]
    n_elec = config.INPUT_SHAPE[1]
    X4d = X_test.reshape(-1, n_freq, n_elec, 2)
    Zre = X4d[:, :, :, 0].mean(axis=2)
    Zim = X4d[:, :, :, 1].mean(axis=2)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in range(config.N_CLASSES):
        mask = y_test_int == cls
        zre_cls, zim_cls = Zre[mask].mean(axis=0), Zim[mask].mean(axis=0)
        ax.plot(zre_cls, -zim_cls, 'o-', color=config.CLASS_COLORS[cls], lw=2, markersize=5,
                label=config.CLASS_NAMES[cls].splitlines()[0])
    ax.set_xlabel("Z' (Real)"); ax.set_ylabel("-Z'' (Imaginary)")
    ax.set_title('Nyquist Plot — Mean per BNP Class', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path or os.path.join(config.FIGURE_DIR, 'nyquist_plot.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
