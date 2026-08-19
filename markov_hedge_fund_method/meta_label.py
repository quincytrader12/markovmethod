"""Meta-labelling — a forest that decides *whether* to take the Markov signal.

The Markov chain already decides direction. Asking a machine-learning model to
also predict direction would be asking it to solve the hard problem with a
thin sample, and it would overfit. López de Prado's meta-labelling splits the
job instead:

    Markov signal says LONG  ──►  forest: "does this particular signal win?"
                                   ──►  size it, or skip it

The forest never flips a trade. It only ever answers a binary question about a
trade the chain has already chosen, which is a far easier problem and one with
a proper label attached.

Three things make or break this, and all three are implemented here:

  * **Triple-barrier labels.** A trade is a win if a profit barrier is touched
    before a stop barrier, within a holding limit — not if the fixed-horizon
    return happens to be positive. Barrier widths scale with the HAR volatility
    forecast, so a 1% move counts as a win in a calm market and as noise in a
    violent one.

  * **Purging and embargo.** Labels overlap in time: with a 5-day horizon,
    consecutive samples share four days of outcome. Training on a sample whose
    outcome window straddles the decision day leaks the future. Every fit here
    drops samples whose label had not yet resolved, plus an embargo buffer.

  * **Sample uniqueness weights.** The same overlap means the effective sample
    is a fraction of the row count — roughly 450 independent observations, not
    2,250. Samples are weighted by how few other samples share their outcome
    days, so a crowded stretch of history does not get counted five times.

The forest itself is deliberately small and written in plain NumPy: shallow
trees, few of them, feature subsampling. That is not a compromise — with a few
hundred effective observations, a deep model is a memoriser. Writing it here
rather than importing scikit-learn also keeps scikit-learn and SciPy out of the
frozen Windows executable, the same reason `sharpe_stats` uses `statistics`
instead of `scipy.stats`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regime import n_step_forecast, signal_confidence, stationary_distribution

TRADING_DAYS = 252

FEATURE_NAMES = [
    "signal",         # P(bull next) - P(bear next), the chain's raw conviction
    "absSignal",      # magnitude alone, so the tree can split on strength
    "z",              # signal / its standard error — is it above the noise?
    "n",              # transitions behind the row; small n, weak evidence
    "daysInRegime",   # how long the current regime has already persisted
    "state",          # 0 bear / 1 sideways / 2 bull
    "pBull5",         # 5-step-ahead bull probability (Chapman-Kolmogorov)
    "pBull20",        # 20-step-ahead — the longer-run drift of the chain
    "volRatio",       # forecast vol / trailing realised vol: calm or heating up
    "rsi14",          # stretched or washed out
    "mom20",          # 20-day momentum
    "maGap",          # close / 50-day mean - 1
]


# ── features ─────────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    d = close.astype(float).diff()
    up = d.clip(lower=0.0).rolling(window).mean()
    dn = (-d.clip(upper=0.0)).rolling(window).mean()
    rs = up / dn.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def feature_frame(close: pd.Series, labels: pd.Series, min_train: int = 252,
                  vol_forecast: pd.Series | None = None) -> pd.DataFrame:
    """The 12 features, computed causally, one row per decision day.

    The Markov features are produced by the same incremental count walk that
    `walk_forward_backtest` runs, so row t holds exactly what the terminal knew
    on day t — nothing later. The price features come from rolling windows,
    which look backwards by construction.
    """
    common = labels.index.intersection(close.index)
    labels = labels.loc[common]
    close = close.loc[common]
    if len(labels) < min_train + 30:
        return pd.DataFrame(columns=FEATURE_NAMES)

    lab = labels.to_numpy().astype(int)
    px = close.astype(float)

    rsi = _rsi(px).to_numpy()
    mom = px.pct_change(20).fillna(0.0).to_numpy()
    ma50 = px.rolling(50).mean()
    gap = (px / ma50 - 1.0).fillna(0.0).to_numpy()
    trail = px.pct_change().rolling(21).std().fillna(0.0).to_numpy() * np.sqrt(TRADING_DAYS)

    vf = None
    if vol_forecast is not None and len(vol_forecast):
        vf = vol_forecast.reindex(labels.index).to_numpy(dtype=float)

    counts = np.zeros((3, 3), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    rows, idx = [], []
    for t in range(min_train, len(lab) - 1):
        if t > min_train:
            counts[lab[t - 2], lab[t - 1]] += 1.0
        state = int(lab[t])
        stats = signal_confidence(counts, state)

        row_tot = counts.sum(axis=1, keepdims=True)
        P = np.divide(counts, row_tot, out=np.full((3, 3), 1 / 3), where=row_tot > 0)
        try:
            p5 = float(n_step_forecast(P, 5)[state, 2])
            p20 = float(n_step_forecast(P, 20)[state, 2])
        except np.linalg.LinAlgError:          # pragma: no cover — degenerate P
            p5 = p20 = 1 / 3

        days = 1
        while days < 60 and t - days >= 0 and lab[t - days] == state:
            days += 1

        sigma = vf[t] if vf is not None and np.isfinite(vf[t]) else np.nan
        base = trail[t] if trail[t] > 0 else np.nan
        if np.isfinite(sigma) and np.isfinite(base) and base > 0:
            vol_ratio = float(sigma / base)
        else:
            vol_ratio = 1.0

        rows.append([
            stats["signal"], abs(stats["signal"]), stats["z"], float(stats["n"]),
            float(days), float(state), p5, p20, vol_ratio,
            float(rsi[t]), float(mom[t]), float(gap[t]),
        ])
        idx.append(labels.index[t])

    return pd.DataFrame(rows, index=pd.Index(idx), columns=FEATURE_NAMES)


def stationary_bull(counts: np.ndarray) -> float:
    """Long-run bull share implied by the counts — kept public for the UI."""
    row_tot = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row_tot, out=np.full((3, 3), 1 / 3), where=row_tot > 0)
    return float(stationary_distribution(P)[2])


# ── triple-barrier labels ────────────────────────────────────────────────────
def triple_barrier(close: pd.Series, side: pd.Series, sigma: pd.Series | None = None,
                   horizon: int = 5, pt: float = 1.0, sl: float = 1.0) -> pd.DataFrame:
    """Label each signal a win (1) or not (0) by which barrier it touches first.

    `side` is the direction the Markov chain chose (+1/-1; 0 days are dropped).
    Barrier half-width is `pt * sigma_daily * sqrt(horizon)` — the typical move
    over the holding period — so the target adapts to the regime's turbulence
    instead of being a fixed percentage that is trivial in a storm and
    unreachable in a lull.

    Returns columns: `label`, `side`, `t1` (integer position where the trade
    resolved) and `ret` (the realised return of the trade).
    """
    px = close.astype(float)
    side = side.reindex(px.index).fillna(0.0)
    if sigma is not None and len(sigma):
        sig_d = (sigma.reindex(px.index).astype(float) / np.sqrt(TRADING_DAYS))
    else:
        sig_d = px.pct_change().rolling(21).std()
    sig_d = sig_d.replace([np.inf, -np.inf], np.nan)
    fallback = float(px.pct_change().std() or 0.01)
    sig_d = sig_d.fillna(fallback).clip(lower=1e-4)

    p = px.to_numpy()
    s = side.to_numpy()
    sd = sig_d.to_numpy()
    n = len(p)

    out_idx, lab, sides, t1s, rets = [], [], [], [], []
    for t in range(n - 1):
        if s[t] == 0:
            continue
        width = float(sd[t]) * np.sqrt(horizon)
        up = p[t] * (1.0 + pt * width)
        dn = p[t] * (1.0 - sl * width)
        stop = min(t + horizon, n - 1)
        hit, end = 0, stop
        for k in range(t + 1, stop + 1):
            if p[k] >= up:
                hit, end = 1 if s[t] > 0 else -1, k
                break
            if p[k] <= dn:
                hit, end = -1 if s[t] > 0 else 1, k
                break
        r = float(s[t] * (p[end] / p[t] - 1.0))
        out_idx.append(px.index[t])
        lab.append(1 if (hit == 1 or (hit == 0 and r > 0)) else 0)
        sides.append(float(s[t]))
        t1s.append(int(end))
        rets.append(r)

    return pd.DataFrame({"label": lab, "side": sides, "t1": t1s, "ret": rets},
                        index=pd.Index(out_idx))


# ── sample uniqueness ────────────────────────────────────────────────────────
def uniqueness_weights(t0: np.ndarray, t1: np.ndarray, n_bars: int) -> np.ndarray:
    """Average uniqueness of each label (AFML 4.1-4.2).

    Count how many labels are live on each bar; a label's weight is the mean of
    1/concurrency over the bars it spans. A sample that had the market to itself
    scores 1.0; one of five overlapping samples scores about 0.2.
    """
    t0 = np.asarray(t0, dtype=int)
    t1 = np.asarray(t1, dtype=int)
    if len(t0) == 0:
        return np.zeros(0)
    conc = np.zeros(n_bars + 1, dtype=float)
    for a, b in zip(t0, t1):
        conc[a: b + 1] += 1.0
    conc[conc == 0] = 1.0
    w = np.empty(len(t0), dtype=float)
    for i, (a, b) in enumerate(zip(t0, t1)):
        w[i] = float(np.mean(1.0 / conc[a: b + 1]))
    return w


def purged_kfold_splits(t0: np.ndarray, t1: np.ndarray, n_splits: int = 5,
                        embargo_pct: float = 0.01) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged k-fold with an embargo (AFML 7.4).

    `t0` and `t1` are the first and last *bar* each sample's outcome spans.
    Test folds are contiguous blocks of samples; a training sample is dropped if
    its outcome window overlaps the fold's bar span at all, and a further
    `embargo_pct` of bars after the fold is withheld to kill the serial
    correlation that would otherwise leak across the boundary. Plain k-fold on
    overlapping financial labels reports skill that does not exist — this is the
    only honest version.
    """
    t0 = np.asarray(t0, dtype=int)
    t1 = np.asarray(t1, dtype=int)
    n = len(t1)
    if n == 0 or n_splits < 2:
        return []
    span = int(t1.max() - t0.min() + 1)
    embargo = int(span * embargo_pct)
    bounds = np.linspace(0, n, n_splits + 1).astype(int)
    splits = []
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        test = np.arange(lo, hi)
        fold_start, fold_end = int(t0[lo]), int(t1[hi - 1])
        overlaps = (t1 >= fold_start) & (t0 <= fold_end)
        embargoed = (t0 > fold_end) & (t0 <= fold_end + embargo)
        keep = np.where(~overlaps & ~embargoed)[0]
        keep = keep[(keep < lo) | (keep >= hi)]
        splits.append((keep, test))
    return splits


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based area under the ROC curve, ties averaged. 0.5 == coin flip."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n1 = float((y == 1).sum())
    n0 = float(len(y) - n1)
    if n0 == 0 or n1 == 0:
        return 0.5
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    sp = p[order]                                  # average the ranks of ties
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i: j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


# ── a small forest, in plain NumPy ───────────────────────────────────────────
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value: float):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = value


class DecisionTree:
    """Weighted CART classifier, gini split, depth- and leaf-limited.

    Split search is vectorised: sort the column once, then evaluate every
    threshold at once with cumulative sums of weight and of weighted positives.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 50,
                 max_features: int | None = 4, rng: np.random.Generator | None = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)
        self.root: _Node | None = None
        self.importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
        self.importances_ = np.zeros(X.shape[1], dtype=float)
        self.root = self._grow(X, y, w, np.arange(len(y)), 0)
        tot = self.importances_.sum()
        if tot > 0:
            self.importances_ /= tot
        return self

    def _leaf(self, y, w, idx) -> _Node:
        wt = w[idx].sum()
        p = float((w[idx] * y[idx]).sum() / wt) if wt > 0 else 0.5
        return _Node(p)

    def _grow(self, X, y, w, idx, depth) -> _Node:
        node = self._leaf(y, w, idx)
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return node
        if len(np.unique(y[idx])) < 2:
            return node

        n_feat = X.shape[1]
        k = n_feat if not self.max_features else min(self.max_features, n_feat)
        cols = self.rng.choice(n_feat, size=k, replace=False)

        best = (0.0, -1, 0.0)                        # (gain, feature, threshold)
        W = w[idx].sum()
        P = (w[idx] * y[idx]).sum()
        parent = 2.0 * P * (W - P) / W if W > 0 else 0.0
        for j in cols:
            v = X[idx, j]
            order = np.argsort(v, kind="stable")
            vs, ys, ws = v[order], y[idx][order], w[idx][order]
            cw = np.cumsum(ws)
            cp = np.cumsum(ws * ys)
            m = self.min_samples_leaf
            if len(idx) <= 2 * m:
                continue
            Wl, Pl = cw[:-1], cp[:-1]
            Wr, Pr = W - Wl, P - Pl
            valid = (vs[:-1] < vs[1:]) & (Wl > 0) & (Wr > 0)
            valid[: m - 1] = False
            valid[len(valid) - m + 1:] = False
            if not valid.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                child = (2.0 * Pl * (Wl - Pl) / Wl + 2.0 * Pr * (Wr - Pr) / Wr) / W
            child = np.where(valid, child, np.inf)
            b = int(np.argmin(child))
            gain = parent - float(child[b])
            if gain > best[0]:
                best = (gain, int(j), float((vs[b] + vs[b + 1]) / 2.0))

        gain, feat, thr = best
        if feat < 0 or gain <= 1e-12:
            return node

        mask = X[idx, feat] <= thr
        left_idx, right_idx = idx[mask], idx[~mask]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return node

        self.importances_[feat] += gain * W
        node.feature, node.threshold = feat, thr
        node.left = self._grow(X, y, w, left_idx, depth + 1)
        node.right = self._grow(X, y, w, right_idx, depth + 1)
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty(len(X), dtype=float)
        for i, row in enumerate(X):
            node = self.root
            while node is not None and node.feature >= 0:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = 0.5 if node is None else node.value
        return out


class RandomForest:
    """Bagged shallow trees. Small on purpose: the effective sample is small."""

    def __init__(self, n_estimators: int = 40, max_depth: int = 3,
                 min_samples_leaf: int = 50, max_features: int = 4, seed: int = 7):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees: list[DecisionTree] = []
        self.importances_ = np.zeros(0)

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "RandomForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
        n = len(y)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        imp = np.zeros(X.shape[1], dtype=float)
        if n < 2 * self.min_samples_leaf:
            return self
        # Bag by *uniqueness weight*, not uniformly: overlapping samples are
        # already near-duplicates, so drawing them proportionally to how much
        # unique information they carry is the sequential-bootstrap idea in its
        # cheap, deterministic form.
        p = w / w.sum() if w.sum() > 0 else None
        for b in range(self.n_estimators):
            pick = rng.choice(n, size=n, replace=True, p=p)
            tree = DecisionTree(self.max_depth, self.min_samples_leaf,
                                self.max_features, np.random.default_rng(self.seed + b))
            tree.fit(X[pick], y[pick], np.ones(n))
            self.trees.append(tree)
            if tree.importances_ is not None:
                imp += tree.importances_
        if imp.sum() > 0:
            imp /= imp.sum()
        self.importances_ = imp
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not self.trees:
            return np.full(len(X), 0.5)
        acc = np.zeros(len(X), dtype=float)
        for tree in self.trees:
            acc += tree.predict_proba(X)
        return acc / len(self.trees)


# ── the walk-forward meta probability series ─────────────────────────────────
def cv_skill(X: np.ndarray, y: np.ndarray, w: np.ndarray, t0: np.ndarray, t1: np.ndarray,
             *, n_splits: int = 5, n_estimators: int = 15, max_depth: int = 3,
             min_samples_leaf: int = 50, embargo_pct: float = 0.01) -> dict:
    """Out-of-sample AUC under purged k-fold — does this model predict at all?

    Fit on the training side of each purged fold, score the held-out side, and
    keep every fold's AUC. 0.5 means the forest is a coin flip on data it has
    not seen, and a coin flip that discards trades is strictly worse than no
    filter at all.

    A single averaged AUC is not enough to act on: a fixed cut-off like 0.53 is
    cleared by chance often enough that, refit after refit, the noise gets
    through and does damage. So the spread *across folds* is returned too, and
    the caller requires the mean to stand clear of 0.5 by more than the sampling
    error of that mean. The bar then scales itself to how noisy this particular
    symbol's estimate is.

    Returns {"auc", "sem", "t", "folds"}.
    """
    splits = purged_kfold_splits(t0, t1, n_splits=n_splits, embargo_pct=embargo_pct)
    flat = {"auc": 0.5, "sem": 0.0, "t": 0.0, "folds": 0}
    if not splits:
        return flat
    scores = []
    for train, test in splits:
        if len(train) < 2 * min_samples_leaf or len(test) < 10:
            continue
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        f = RandomForest(n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf).fit(X[train], y[train], w[train])
        if not f.trees:
            continue
        scores.append(auc(y[test], f.predict_proba(X[test])))
    if len(scores) < 2:
        return flat
    a = np.array(scores, dtype=float)
    sem = float(a.std(ddof=1) / np.sqrt(len(a)))
    return {"auc": float(a.mean()), "sem": sem,
            "t": float((a.mean() - 0.5) / sem) if sem > 0 else 0.0,
            "folds": len(a)}


def meta_probabilities(close: pd.Series, labels: pd.Series, *, min_train: int = 252,
                       vol_forecast: pd.Series | None = None, horizon: int = 5,
                       warmup: int = 252, refit_every: int = 63, embargo: int = 5,
                       n_estimators: int = 40, max_depth: int = 3,
                       min_samples_leaf: int = 50, min_auc: float = 0.55,
                       min_t: float = 3.0) -> dict:
    """P(this signal wins), day by day, fit only on the past.

    The forest is refit quarterly on every sample whose triple-barrier outcome
    had already resolved `embargo` days before the decision day.

    Every refit has to earn the right to filter. The same training data is first
    run through purged k-fold cross-validation, and the model is only used if
    its out-of-sample AUC clears `min_auc` *and* stands `min_t` standard errors
    clear of a coin flip. Otherwise it is set aside and the layer passes every
    signal through untouched until the next refit.

    That gate is not decoration. Measured on eight structureless simulations,
    an ungated forest cut the Sharpe on seven of them: a model with no skill
    that still rejects half the days is not neutral, it is a random
    trade-destroyer. With a fixed AUC cut-off the noise still slipped through
    often enough to cost 0.09 Sharpe at worst. At the defaults here the same
    eight runs come out at -0.004 Sharpe at worst — effectively a no-op — while
    a simulation with a genuine planted relationship still engages on five runs
    out of six. Harmless when there is nothing to find is the property that
    makes the layer safe to leave switched on.

    Returns {"prob", "importances", "nTrain", "threshold", "cvAuc", "active"}.
    """
    idle = {"prob": pd.Series(dtype=float), "importances": {}, "nTrain": 0,
            "threshold": 0.5, "cvAuc": 0.5, "active": False}
    feats = feature_frame(close, labels, min_train=min_train, vol_forecast=vol_forecast)
    if feats.empty:
        return idle

    side = np.sign(feats["signal"].to_numpy())
    side_s = pd.Series(side, index=feats.index)
    px = close.reindex(feats.index)
    bars = triple_barrier(px, side_s, sigma=vol_forecast, horizon=horizon)
    if bars.empty:
        return idle

    pos = {d: i for i, d in enumerate(feats.index)}
    t0 = np.array([pos[d] for d in bars.index], dtype=int)
    t1 = bars["t1"].to_numpy().astype(int)
    y = bars["label"].to_numpy().astype(float)
    X = feats.loc[bars.index].to_numpy(dtype=float)
    w = uniqueness_weights(t0, t1, len(feats))

    forest: RandomForest | None = None
    probs, out_idx = [], []
    n_train_last, cv_last, thr = 0, 0.5, 0.5
    ever_active = False
    fitted_any = False
    for i, t in enumerate(t0):
        if t < warmup:
            continue
        if not fitted_any or (t - warmup) % refit_every == 0:
            # purge: only outcomes that had fully resolved, plus the embargo
            avail = np.where(t1 <= t - embargo)[0]
            if len(avail) >= max(4 * min_samples_leaf, 100):
                fitted_any = True
                cv = cv_skill(X[avail], y[avail], w[avail], t0[avail], t1[avail],
                              max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                cv_last = cv["auc"]
                if cv["auc"] >= min_auc and cv["t"] >= min_t:
                    forest = RandomForest(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf).fit(
                        X[avail], y[avail], w[avail])
                    if not ever_active:
                        # "Better than average" is the bar. Fixed at the first
                        # live fit so one threshold governs the whole curve
                        # instead of drifting quarter to quarter.
                        thr = float(np.average(y[avail], weights=w[avail]))
                    n_train_last = len(avail)
                    ever_active = True
                else:
                    forest = None                  # no demonstrated skill: stand down
        if not fitted_any:
            continue
        # A quarter with no skill passes everything through, so the meta curve
        # is the base curve rather than a randomly thinned version of it.
        p = 1.0 if forest is None or not forest.trees else float(
            forest.predict_proba(X[i: i + 1])[0])
        probs.append(p)
        out_idx.append(bars.index[i])

    imp = {}
    if forest is not None and len(forest.importances_) == len(FEATURE_NAMES):
        imp = {name: float(v) for name, v in zip(FEATURE_NAMES, forest.importances_)}
    return {"prob": pd.Series(probs, index=pd.Index(out_idx), name="metaProb"),
            "importances": imp, "nTrain": int(n_train_last),
            "threshold": float(thr), "cvAuc": float(cv_last),
            "active": bool(ever_active)}
