import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_precision(pred, true):
    correct_num = (pred & true).sum()
    return correct_num / (pred.sum() + 1e-8)


def get_recall(pred, true):
    correct_num = (pred & true).sum()
    return correct_num / (true.sum() + 1e-8)


def get_f1(pred, true):
    precision = get_precision(pred, true)
    recall = get_recall(pred, true)
    return 2 * precision * recall / (precision + recall + 1e-8)


def adjust_predicts(pred, true, min_ratio=0):
    """
    Adjust predicted results by groud truth.

    Returns:
        - pred: adjusted pred.
    """ 
    def _get_anomaly_interval():
        true_diff = np.diff(true, prepend=[0], append=[0])
        low = np.argwhere(true_diff > 0).reshape(-1)
        high = np.argwhere(true_diff < 0).reshape(-1)
        return low, high

    for low, high in zip(*_get_anomaly_interval()):
        if pred[low: high].sum() >= max(1, min_ratio * (high - low)):
            pred[low: high] = 1

    return pred


def get_best_threshold(scores, true, iter_steps=1000, adjust=True, min_ratio=0):
    """
    Get threshold of anomaly scores corresponding to the best f1.

    Returns:
        - threshold: Best threshold.
        - f1: best f1 score.
    """
    sorted_index = np.argsort(scores)
    th_vals = np.linspace(0, len(scores) - 1, num=iter_steps)

    best_f1, best_thr = 0, 0
    for th_val in th_vals:
        cur_thr = scores[sorted_index[int(th_val)]]
        cur_pred = (scores >= cur_thr).astype(int)
        if adjust:
            cur_pred = adjust_predicts(cur_pred, true, min_ratio)
        cur_f1 = get_f1(cur_pred, true)

        if cur_f1 > best_f1:
            best_f1, best_thr = cur_f1, cur_thr

    return best_thr, best_f1


def get_pak_auc(scores, true, ratios=None):
    def _func(pos, ratio):
        _, f1 = get_best_threshold(scores, true, min_ratio=ratio)
        return f1, pos

    if ratios is None:
        ratios = [0, 0.01, 0.02, 0.05] + list(np.linspace(0.1, 1, 10))
    results, tasks = np.zeros(len(ratios)), []

    with ThreadPoolExecutor(max_workers=6) as exec:
        for i, t in enumerate(ratios):
            tasks.append(exec.submit(_func, i, t))

    for task in as_completed(tasks):
        f1, pos = task.result()
        results[pos] = f1

    return results
