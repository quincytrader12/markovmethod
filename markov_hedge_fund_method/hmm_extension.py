"""Optional Hidden Markov Model layer. Imports hmmlearn lazily so the
observable model still works if hmmlearn failed to install."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_hmm(returns: pd.Series, n_components: int = 3, random_state: int = 42):
    """Fit a Gaussian HMM on daily returns. Returns (model, hidden_states).

    Caveat: Baum-Welch finds local maxima. For production work, fit with
    several random_state values and keep the best by log-likelihood.
    """
    try:
        from hmmlearn import hmm  # lazy import
    except ImportError:
        return None, None

    X = returns.dropna().to_numpy().reshape(-1, 1)
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(X)
    hidden_states = model.predict(X)
    return model, hidden_states
