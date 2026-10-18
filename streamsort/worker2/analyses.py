"""Per-trial analyses (swappable): Procrustes geometry against a target, or the placeholder image decoder.
Both take X = pooled binned rates (n_bins, n_units) and the trial's condition id, and return
(loss or None while fitting, display dict)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry.core import make_reducer, procrustes_distance


class ProcrustesAnalysis:
    """Reducer (pca | fa | isomap) fitted causally on condition means of the first n_fit trials.
    Target = mean trajectory of `target_condition` over those trials (or a loaded (n_bins, k) array).
    Loss = Procrustes distance (scaled) of the trial trajectory to the target."""
    name = "procrustes"

    def __init__(self, reducer="pca", k=3, n_fit=20, target_condition=None, target=None, **_):
        self.red_name, self.k, self.n_fit, self.tc = reducer, int(k), int(n_fit), target_condition
        self.target0 = None if target is None else np.asarray(np.load(target) if isinstance(target, str) else target, float)
        self.reset()

    def reset(self):
        self.red, self.target, self.fit_X, self.fit_c = None, self.target0, [], []

    @property
    def ready(self):
        return self.red is not None

    def __call__(self, X, cond):
        if self.red is None:
            self.fit_X.append(X); self.fit_c.append(cond)
            if len(self.fit_X) < self.n_fit:
                return None, dict(state="fitting", n=len(self.fit_X), n_fit=self.n_fit)
            c = np.array([str(x) for x in self.fit_c]); Xs = np.stack(self.fit_X)
            self.red = make_reducer(self.red_name, self.k).fit(np.concatenate([Xs[c == u].mean(0) for u in np.unique(c)]))
            if self.target is None:
                sel = (c == str(self.tc)) if self.tc is not None and (c == str(self.tc)).any() else np.ones(len(c), bool)
                self.target = self.red.transform(Xs[sel].mean(0))
            return None, dict(state="armed", target=self.target)
        z = self.red.transform(X)
        d = float(procrustes_distance(self.target, z))
        # aligned trial for display: orthogonal Procrustes of z onto the target (centred, scaled)
        A = self.target - self.target.mean(0); B = z - z.mean(0)
        u, _, vt = np.linalg.svd(B.T @ A); R = u @ vt
        Bn = B @ R; s = np.linalg.norm(A) / max(np.linalg.norm(Bn), 1e-12)
        return d, dict(state="running", target=self.target, trial=Bn * s + self.target.mean(0), raw=z)


class ImageDecoderAnalysis:
    """Placeholder decoder over a fixed set of presented images.
    Classes = condition ids; each class has a PNG (`images[class]`). Without a model file, a
    nearest-centroid readout on mean rates is fitted causally on the first n_fit trials.
    With a model npz (W (n_units, n_classes), b, classes) that readout is used as given.
    Output: decoded class -> its PNG; loss = -log p(target class) (softmax of -distance / scores)."""
    name = "image_decoder"

    def __init__(self, images: dict | None = None, target_class=None, n_fit=20, model=None, **_):
        self.images = {int(k): str(v) for k, v in (images or {}).items()}
        self.target_class = None if target_class is None else int(target_class)
        self.n_fit, self.model = int(n_fit), model
        self.reset()

    def reset(self):
        self.cent, self.classes, self.fit_X, self.fit_c = None, None, [], []
        if self.model:
            m = np.load(self.model)
            self.W, self.b, self.classes = m["W"], m["b"], [int(c) for c in m["classes"]]

    @property
    def ready(self):
        return self.cent is not None or self.model is not None

    def _probs(self, r):
        if self.model:
            a = r @ self.W + self.b
        else:
            a = -np.linalg.norm(self.cent - r[None], axis=1) / max(self.scale, 1e-12)
        a = a - a.max(); p = np.exp(a); return p / p.sum()

    def __call__(self, X, cond):
        r = X.mean(0)
        if not self.ready:
            self.fit_X.append(r); self.fit_c.append(int(cond) if cond is not None else -1)
            if len(self.fit_X) < self.n_fit:
                return None, dict(state="fitting", n=len(self.fit_X), n_fit=self.n_fit, target_image=self.images.get(self.target_class))
            R = np.stack(self.fit_X); c = np.array(self.fit_c)
            self.classes = sorted(set(c.tolist()) - {-1})
            self.cent = np.stack([R[c == k].mean(0) for k in self.classes])
            self.scale = float(np.median([np.linalg.norm(R[i] - self.cent[self.classes.index(c[i])]) for i in range(len(c)) if c[i] in self.classes]) or 1.0)
            return None, dict(state="armed", target_image=self.images.get(self.target_class))
        if len(r) != (self.W.shape[0] if self.model else self.cent.shape[1]):
            return None, dict(state="unit set changed")
        p = self._probs(r); k = int(np.argmax(p)); dec = self.classes[k]
        tgt = self.target_class if self.target_class in self.classes else None
        loss = float(-np.log(max(p[self.classes.index(tgt)], 1e-12))) if tgt is not None else float(-np.log(max(p[k], 1e-12)))
        return loss, dict(state="running", probs=p, classes=list(self.classes), decoded=dec,
                          decoded_image=self.images.get(dec), target_image=self.images.get(tgt), true_class=cond)


def export_linear_readout(an: "ImageDecoderAnalysis", path) -> Path:
    """Save a fitted nearest-centroid decoder as an equivalent-form linear readout npz
    (equal-variance Gaussian: W = c_k / s^2, b = -|c_k|^2 / (2 s^2)); loadable as `model=`."""
    if an.cent is None:
        raise ValueError("decoder not fitted yet")
    s2 = max(an.scale, 1e-12) ** 2
    W = (an.cent / s2).T; b = -(an.cent ** 2).sum(1) / (2 * s2)
    np.savez(path, W=W.astype(np.float32), b=b.astype(np.float32), classes=np.asarray(an.classes, np.int64))
    return Path(path)


ANALYSES = {"procrustes": ProcrustesAnalysis, "image_decoder": ImageDecoderAnalysis}


def make_placeholder_images(n: int, folder, size: int = 96) -> dict:
    """One PNG per presented image: oriented gratings at n orientations (placeholder stimulus set)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size] / size
    out = {}
    for k in range(n):
        th = np.pi * k / n; sf = 4 + (k % 3) * 2
        img = 0.5 + 0.5 * np.cos(2 * np.pi * sf * (xx * np.cos(th) + yy * np.sin(th)))
        p = folder / f"image_{k:02d}.png"; plt.imsave(p, img, cmap="gray", vmin=0, vmax=1); out[k] = str(p)
    return out
