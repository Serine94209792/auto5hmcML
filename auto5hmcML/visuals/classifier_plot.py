import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline

def plot_calibration_curve(model:Pipeline,
                           X: pd.DataFrame,
                           y: pd.Series,
                           n_bins=10,
                           show_histogram=True)->None:
    """
    Plot a calibration curve for a classification model.

    Parameters
    ----------
    n_bins : int, optional (default=10)
        Number of bins to discretize the [0,1] interval.
    show_histogram : bool, optional (default=False)
        If True, plot a histogram of predicted probabilities.
    """

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-scores))  # Sigmoid
    else:
        raise ValueError("Model must implement predict_proba or decision_function")

    fraction_of_positives, mean_predicted_value = calibration_curve(y, y_prob, n_bins=n_bins)

    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig("calibration_curve.png", dpi=300)

    # Optionally, plot histogram of predicted probabilities
    if show_histogram:
        plt.hist(y_prob, range=(0, 1), bins=n_bins, histtype="step", lw=1)
        plt.title("predicted probabilities histogram")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("histogram.png", dpi=300)

def plot_ks_curve(model:Pipeline,
                  X: pd.DataFrame,
                  y: pd.Series)->None:
    """
    Plot a KS curve for a binary classification model.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-scores))  # Sigmoid
    else:
        raise ValueError("Model must implement predict_proba or decision_function")

    # Create a DataFrame and sort by predicted probability
    df = pd.DataFrame({'y_true': y, 'y_prob': y_prob})
    df = df.sort_values(by='y_prob')

    # Calculate cumulative distributions
    total_pos = (df['y_true'] == 1).sum()
    total_neg = (df['y_true'] == 0).sum()
    df['cum_pos'] = (df['y_true'] == 1).cumsum() / total_pos
    df['cum_neg'] = (df['y_true'] == 0).cumsum() / total_neg

    # Compute KS statistic
    df['diff'] = np.abs(df['cum_pos'] - df['cum_neg'])
    ks_stat = df['diff'].max()
    ks_idx = df['diff'].idxmax()

    # Plot KS curve
    plt.figure()
    plt.plot(df['y_prob'], df['cum_pos'], label='Cumulative Positive')
    plt.plot(df['y_prob'], df['cum_neg'], label='Cumulative Negative')
    # Highlight KS
    plt.vlines(df.loc[ks_idx, 'y_prob'],
               df.loc[ks_idx, 'cum_neg'],
               df.loc[ks_idx, 'cum_pos'],
               color='red', linestyle='--', label=f'KS = {ks_stat:.3f}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Cumulative Distribution')
    plt.title('Kolmogorov-Smirnov Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("ks_curve.png", dpi=300)

def plot_cumulative_gain(model:Pipeline,
                  X: pd.DataFrame,
                  y: pd.Series)->None:
    """
    Plot a cumulative gains curve for a binary classification model.
    """

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        raise ValueError("Model must implement predict_proba or decision_function")

    df = pd.DataFrame({'y_true': y, 'y_prob': y_prob})
    df = df.sort_values(by='y_prob', ascending=False).reset_index(drop=True)


    df['cum_pos'] = (df['y_true'] == 1).cumsum()
    total_pos = df['y_true'].sum()
    df['gain'] = df['cum_pos'] / total_pos

    df['pct_samples'] = (df.index + 1) / len(df)

    plt.figure()
    plt.plot(df['pct_samples'], df['gain'], linewidth=2, label='Model Gain')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline', color='gray')
    plt.xlabel('Fraction of Samples')
    plt.ylabel('Fraction of Positives Captured')
    plt.title('Cumulative Gains Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("cumulative_gain_curve.png", dpi=300)


def plot_lift_curve(model:Pipeline,
                  X: pd.DataFrame,
                  y: pd.Series)->None:
    """
    Plot a lift curve for a binary classification model.
    """
    # Obtain predicted probabilities for positive class
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-scores))
    else:
        raise ValueError("Model must implement predict_proba or decision_function")

    df = pd.DataFrame({'y_true': y, 'y_prob': y_prob})
    df = df.sort_values(by='y_prob', ascending=False).reset_index(drop=True)

    df['cum_pos'] = (df['y_true'] == 1).cumsum()
    total_pos = df['y_true'].sum()
    df['gain'] = df['cum_pos'] / total_pos
    df['pct_samples'] = (df.index + 1) / len(df)

    df['lift'] = df['gain'] / df['pct_samples']

    plt.figure()
    plt.plot(df['pct_samples'], df['lift'], linewidth=2, label='Model Lift')
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Baseline (Lift = 1)')
    plt.xlabel('Fraction of Samples')
    plt.ylabel('Lift = Gain / Fraction of Samples')
    plt.title('Lift Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("lift_curve.png", dpi=300)
