from typing import Optional, Sequence


def accuracy(gold: Sequence[str], pred: Sequence[Optional[str]]) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def macro_precision(gold: Sequence[str], pred: Sequence[Optional[str]]) -> float:
    """Unweighted mean over the classes in `gold` of: correct predictions of the class / all predictions of it.

    A class the model never predicted scores 0. A failed call (None) is not a prediction of any class.
    """
    scores = []
    for cls in sorted(set(gold)):
        predicted = sum(p == cls for p in pred)
        correct = sum(g == cls and p == cls for g, p in zip(gold, pred))
        scores.append(correct / predicted if predicted else 0.0)
    return sum(scores) / len(scores)


def macro_recall(gold: Sequence[str], pred: Sequence[Optional[str]]) -> float:
    """Unweighted mean over the classes in `gold` of: correct predictions of the class / all samples of it.

    A failed call (None) counts as a miss for the gold class.
    """
    scores = []
    for cls in sorted(set(gold)):
        actual = sum(g == cls for g in gold)
        correct = sum(g == cls and p == cls for g, p in zip(gold, pred))
        scores.append(correct / actual)
    return sum(scores) / len(scores)
