
from sklearn.metrics import average_precision_score
from torch.nn.functional import one_hot


def avg_precision(y_pred, y_true, n_classes):
    assert y_pred.shape[1] == n_classes, y_pred.shape
    return average_precision_score(one_hot(y_true, n_classes), y_pred)
