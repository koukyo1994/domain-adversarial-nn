import numpy as np

from tqdm import tqdm


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = matthews_correlation(
            y_true.astype(np.float64),
            (y_proba > threshold).astype(np.float64))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}
    return search_result


def matthews_correlation(y_true: np.ndarray, y_pred: np.ndarray):
    y_pred_pos = (y_pred > 0.5).astype(float)
    y_pred_neg = 1 - y_pred_pos

    y_pos = (y_true > 0.5).astype(float)
    y_neg = 1 - y_pos

    tp = (y_pos * y_pred_pos).sum()
    tn = (y_neg * y_pred_neg).sum()

    fp = (y_neg * y_pred_pos).sum()
    fn = (y_pos * y_pred_neg).sum()

    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + 1e-8)
