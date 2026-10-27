"""The adopted scorer (Stage 6d): logistic regression on the DINOv2 CLS embedding, trained on the owner-labeled TRAIN shelf entries.

Classes: the 6 products + `unknown`. scikit-learn fits it (L2, multinomial, class_weight balanced, standardised features); the fitted model is
kept as plain arrays (standardisation, weights, intercept) and saved as an .npz next to a small model card, so predicting needs only NumPy
and no pickled object. Probabilities come out of a softmax over the 7 classes.
"""
import numpy as np


class LinearClassifier:
    def __init__(self, classes: list[str], mu, sd, coef, intercept, c: float):
        self.classes, self.mu, self.sd, self.coef, self.intercept, self.c = list(classes), mu, sd, coef, intercept, c

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, classes: list[str], c: float, max_iter: int = 2000):
        """`y` are indices into `classes`; every class must be present in the training data."""
        from sklearn.linear_model import LogisticRegression  # imported here: only fitting needs scikit-learn
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-6
        m = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=max_iter, tol=1e-10)
        with np.errstate(all="ignore"):   # spurious "matmul" warnings of NumPy on Apple Accelerate (see identify.cosine_matrix); results are asserted finite
            m.fit((x - mu) / sd, y)
        assert list(m.classes_) == list(range(len(classes))), "every class must be present in the training data"
        assert np.isfinite(m.coef_).all() and np.isfinite(m.intercept_).all()
        return cls(classes, mu, sd, m.coef_.T.copy(), m.intercept_.copy(), c)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            z = ((x - self.mu) / self.sd) @ self.coef + self.intercept
        assert np.isfinite(z).all()
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path) -> None:
        np.savez(path, classes=np.array(self.classes), mu=self.mu, sd=self.sd, coef=self.coef, intercept=self.intercept, c=self.c)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls([str(c) for c in z["classes"]], z["mu"], z["sd"], z["coef"], z["intercept"], float(z["c"]))
