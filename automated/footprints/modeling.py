"""Classifier, cross-validated probabilities, threshold selection, curves.

Thresholds are chosen on out-of-fold probabilities rather than in-sample ones,
so the operating point is not set on data the model has already fitted. The
plateau rule takes the highest threshold within --plateau-tol of the peak
F-beta, restricted to the contiguous run of thresholds containing that peak.

Importing this module imports pyplot, so an application that needs a
non-interactive backend must call matplotlib.use() before importing it.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as skmetrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from .util import log, warn


def make_model(args):
    """Instantiate the classifier. No feature scaling, as in the notebooks."""
    if args.model == 'logreg':
        return LogisticRegression(
            max_iter=args.max_iter,
            class_weight=None if args.class_weight == 'none' else 'balanced')
    layers = tuple(int(s) for s in args.hidden_layers.split(','))
    return MLPClassifier(hidden_layer_sizes=layers, max_iter=args.max_iter,
                         n_iter_no_change=40, random_state=args.seed)


def _fit(model, X, y):
    """Fit, converting warnings about non-convergence into recorded warnings."""
    import warnings as _warnings
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X, y)
        for c in caught:
            if issubclass(c.category, ConvergenceWarning):
                warn(f'Solver did not converge: {c.message}')
    return model


def oof_probabilities(X, y, args):
    """Cross-validated out-of-fold positive-class probabilities.

    With ~1k positives, out-of-fold probabilities over the whole labeled set are
    a better use of the labels than a single 80/20 split, which would both bias
    the reported metrics (the threshold gets picked on the same points it is
    scored on) and be high-variance.
    """
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.seed)
    oof = np.zeros(len(y), dtype='f8')
    fold_stats = []
    for k, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = _fit(make_model(args), X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof[te] = p
        fold_stats.append({
            'fold': k,
            'n_train': int(len(tr)),
            'n_test': int(len(te)),
            'average_precision': float(
                skmetrics.average_precision_score(y[te], p)),
            'roc_auc': float(skmetrics.roc_auc_score(y[te], p)),
        })
        log(f'  fold {k}/{args.cv_folds}: AP={fold_stats[-1]["average_precision"]:.4f} '
            f'ROC-AUC={fold_stats[-1]["roc_auc"]:.4f}')
    return oof, fold_stats


def fbeta_from_pr(precision, recall, beta):
    """F-beta from precision/recall arrays, guarding the 0/0 case."""
    b2 = beta ** 2
    num = (1 + b2) * precision * recall
    den = b2 * precision + recall
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def select_threshold(y_true, probs, beta, plateau_tol):
    """Pick a threshold at the rightmost edge of the F-beta curve's peak plateau.

    The plateau is the *contiguous* run of thresholds whose F-beta is within
    plateau_tol of the maximum **and which contains the maximum**. Taking its
    highest threshold is the automated form of 'the rightmost edge of the
    central plateau', and favours precision, hence tighter footprints.
    plateau_tol=0 collapses this to the literal argmax.

    The contiguity requirement matters: on a ragged curve (a weak model) the
    within-tolerance set can break into several runs separated by dips, and
    picking the rightmost point overall would jump into a disconnected bump that
    has nothing to do with the peak. Observed on New Mexico, where the eligible
    set split into [0.238..0.492] (containing the peak) and [0.502..0.624], and
    the old rule selected 0.624.
    """
    precision, recall, thresholds = skmetrics.precision_recall_curve(
        y_true, probs)
    fbeta = fbeta_from_pr(precision[:-1], recall[:-1], beta)
    if len(thresholds) == 0:
        raise SystemExit('Degenerate probabilities; cannot pick a threshold.')

    best = float(fbeta.max())
    argmax = int(np.argmax(fbeta))
    eligible = np.flatnonzero(fbeta >= best - plateau_tol)

    # Split the eligible indices into contiguous runs, keep the one holding the
    # peak, and take its highest threshold.
    runs = np.split(eligible, np.flatnonzero(np.diff(eligible) > 1) + 1)
    plateau = next(r for r in runs if r[0] <= argmax <= r[-1])
    pick = int(plateau[-1])

    return {
        'threshold': float(thresholds[pick]),
        'fbeta_at_threshold': float(fbeta[pick]),
        'fbeta_max': best,
        'argmax_threshold': float(thresholds[argmax]),
        'n_plateau_candidates': int(len(plateau)),
        'n_eligible_total': int(len(eligible)),
        'n_eligible_runs': int(len(runs)),
        # Runs of a single candidate are artefacts of saturated probabilities
        # (Kansas shows two such singletons near zero); only multi-point runs
        # indicate a genuinely ragged curve.
        'n_substantive_runs': int(sum(1 for r in runs if len(r) >= 2)),
        'plateau_threshold_range': [float(thresholds[plateau[0]]),
                                    float(thresholds[plateau[-1]])],
        'precision_at_threshold': float(precision[pick]),
        'recall_at_threshold': float(recall[pick]),
    }


def fbeta_curve_fig(y_true, probs, beta, threshold):
    """F-beta against threshold, with the selected threshold marked.

    Two panels: a linear threshold axis as in the notebooks, and a logit axis.
    A well-separated linear probe saturates, so the linear curve is a flat
    plateau from ~0.01 to ~0.99 and hides the region the threshold is actually
    chosen in; the logit panel resolves 0.9 / 0.99 / 0.999 / 0.9999.
    """
    linear = np.linspace(0, 1, 401)
    # Denser sampling towards 1, where a saturated classifier does its work.
    logit_grid = np.unique(np.concatenate([
        1 - np.logspace(-8, np.log10(0.5), 200), np.logspace(-8, np.log10(0.5),
                                                             200)]))

    def curve(grid):
        return [skmetrics.fbeta_score(y_true, (probs >= t).astype(int),
                                      beta=beta, zero_division=0) for t in grid]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, grid, scale, title in (
            (axes[0], linear, 'linear', 'Linear threshold axis'),
            (axes[1], logit_grid, 'logit', 'Logit threshold axis')):
        ax.plot(grid, curve(grid),
                label=f'F{beta:g} (out-of-fold, patchwise)')
        if 0 < threshold < 1 or scale == 'linear':
            ax.axvline(min(max(threshold, 1e-8), 1 - 1e-9), color='k',
                       ls='--', lw=1,
                       label=f'selected threshold = {threshold:.6g}')
        if scale == 'logit':
            ax.set_xscale('logit')
        ax.set_xlabel('Threshold')
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.legend(loc='lower left', fontsize=8)
    axes[0].set_ylabel(f'F{beta:g} score')
    fig.tight_layout()
    return fig


def pr_curve_fig(y_true, probs):
    """Precision-recall curve from out-of-fold probabilities."""
    precision, recall, _ = skmetrics.precision_recall_curve(y_true, probs)
    ap = skmetrics.average_precision_score(y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f'out-of-fold (AP = {ap:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    return fig
