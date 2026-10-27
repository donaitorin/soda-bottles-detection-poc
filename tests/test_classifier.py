import numpy as np

from src.embeddings.classifier import LinearClassifier

CLASSES = ["a", "b", "unknown"]


def _data():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (60, 5)) + [4, 0, 0, 0, 0], rng.normal(0, 1, (60, 5)) + [0, 4, 0, 0, 0], rng.normal(0, 1, (6, 5)) + [0, 0, 4, 0, 0]])
    return x, np.array([0] * 60 + [1] * 60 + [2] * 6)


def test_fit_predicts_separable_classes_and_does_not_swallow_the_rare_one():
    x, y = _data()
    m = LinearClassifier.fit(x, y, CLASSES, c=1.0)
    p = m.predict_proba(x)
    assert np.allclose(p.sum(axis=1), 1) and (p.argmax(axis=1) == y).mean() > 0.95
    assert m.predict_proba(x[-6:]).argmax(axis=1).tolist() == [2] * 6      # class_weight balanced keeps the 6-crop class


def test_save_and_load_give_the_same_probabilities(tmp_path):
    x, y = _data()
    m = LinearClassifier.fit(x, y, CLASSES, c=0.5)
    m.save(tmp_path / "m.npz")
    m2 = LinearClassifier.load(tmp_path / "m.npz")
    assert m2.classes == CLASSES and m2.c == 0.5 and np.allclose(m.predict_proba(x), m2.predict_proba(x))


def test_a_missing_class_is_refused():
    import pytest
    x, y = _data()
    with pytest.raises(AssertionError):
        LinearClassifier.fit(x[:120], y[:120], CLASSES, c=1.0)
