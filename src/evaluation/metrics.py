from typing import Optional, Sequence


def accuracy(gold: Sequence[str], pred: Sequence[Optional[str]]) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def macro_f1(gold: Sequence[str], pred: Sequence[Optional[str]]) -> float:
    """Unweighted mean of per-class F1 over the classes present in `gold`.

    A prediction outside those classes (or None for a failed call) counts as a miss for
    the gold class, and never adds a class of its own.
    """
    scores = []
    for cls in sorted(set(gold)):
        tp = sum(g == cls and p == cls for g, p in zip(gold, pred))
        fp = sum(g != cls and p == cls for g, p in zip(gold, pred))
        fn = sum(g == cls and p != cls for g, p in zip(gold, pred))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores)
