"""Gaussian HMM on continuous pair variables (v5 notebook ``20261108_KP_pair_hmm_v5.ipynb``).

Alternative to the binned grid state of ``pair_state4d``: the 14 pair variables
``(D_xy, dz, theta_a, theta_b, speed_a, speed_b, bearing_rate, approach, turn_a, turn_b,
bend_a, bend_b, pitch_a, pitch_b)`` are turned into 16 roughly Gaussian features and a
Gaussian HMM is fitted on the training videos without labels. The decoded hidden state of
every frame then plays the role of a microstate (scored, or clustered with Infomap).

Feature map (column order ``FEATURES``): ``log1p`` of distances, signed ``log1p`` of signed
quantities, ``cos`` / ``sin`` of the headings, ``log10`` of speeds and rates, raw bend and pitch.
Contact frames (angles undefined) get ``cos = sin = 0`` (no real heading has that) and the
training median of each undefined rate, so they stay in the sequence instead of being dropped.
"""

from __future__ import annotations

import numpy as np

try:  # import torch before scikit-learn / hmmlearn: on Windows a later torch import can fail (WinError 1114)
    import torch as _torch  # noqa: F401
except ImportError:  # the HMM features and hmmlearn path still work without torch
    _torch = None

VARIABLES = ("D_xy", "dz", "theta_a", "theta_b", "speed_a", "speed_b", "bearing_rate", "approach",
             "turn_a", "turn_b", "bend_a", "bend_b", "pitch_a", "pitch_b")
FEATURES = ("log1p_D_xy", "slog_dz", "cos_theta_a", "sin_theta_a", "cos_theta_b", "sin_theta_b",
            "log_speed_a", "log_speed_b", "log_bearing_rate", "slog_approach", "log_turn_a", "log_turn_b",
            "bend_a", "bend_b", "pitch_a", "pitch_b")
RATE_FLOOR = 0.01    # rad/s added before log10 of angular rates
SPEED_FLOOR = 1.0    # mm/s added before log10 of speeds
ANGLE_RATE_FEATURES = ("log_bearing_rate", "log_turn_a", "log_turn_b")


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def raw_features(values: np.ndarray) -> np.ndarray:
    """``(T, 14)`` variables (order ``VARIABLES``) -> ``(T, 16)`` features (order ``FEATURES``); NaN stays NaN."""
    v = np.asarray(values, dtype=float)
    if v.shape[1] != len(VARIABLES):
        raise ValueError(f"expected {len(VARIABLES)} columns")
    with np.errstate(invalid="ignore"):
        return np.column_stack([
            np.log1p(v[:, 0]), _slog(v[:, 1]),
            np.cos(v[:, 2]), np.sin(v[:, 2]), np.cos(v[:, 3]), np.sin(v[:, 3]),
            np.log10(v[:, 4] + SPEED_FLOOR), np.log10(v[:, 5] + SPEED_FLOOR),
            np.log10(v[:, 6] + RATE_FLOOR), _slog(v[:, 7]),
            np.log10(v[:, 8] + RATE_FLOOR), np.log10(v[:, 9] + RATE_FLOOR),
            v[:, 10], v[:, 11], v[:, 12], v[:, 13]])


def fill_contact(features: np.ndarray, contact: np.ndarray, rate_fill: dict) -> np.ndarray:
    """Contact frames: heading features -> 0, undefined angular rates -> ``rate_fill[name]``."""
    f = np.array(features, dtype=float, copy=True)
    c = np.asarray(contact, dtype=bool)
    for name in ("cos_theta_a", "sin_theta_a", "cos_theta_b", "sin_theta_b"):
        f[c, FEATURES.index(name)] = 0.0
    for name in ANGLE_RATE_FEATURES:
        j = FEATURES.index(name)
        f[c & ~np.isfinite(f[:, j]), j] = rate_fill[name]
    return f


def standardise(features: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (features - mean) / np.where(sd > 0, sd, 1.0)


def runs(mask: np.ndarray, block: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) of runs of True; a change of ``block`` also ends a run."""
    mask = np.asarray(mask, dtype=bool)
    brk = np.ones(len(mask), dtype=bool)
    brk[1:] = mask[1:] != mask[:-1]
    if block is not None:
        brk[1:] |= np.asarray(block)[1:] != np.asarray(block)[:-1]
    starts = np.flatnonzero(brk)
    ends = np.concatenate([starts[1:], [len(mask)]])
    keep = mask[starts]
    return starts[keep], ends[keep]


def fit_hmm(X: np.ndarray, lengths: np.ndarray, n_states: int, seed: int, n_iter: int = 60,
            tol_per_frame: float = 1e-4, covariance_type: str = "full"):
    """Gaussian HMM (hmmlearn) fitted by EM on sequences ``X`` (concatenated) with ``lengths``.
    Convergence: log-likelihood gain below ``tol_per_frame`` x number of frames."""
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(n_components=int(n_states), covariance_type=covariance_type, n_iter=int(n_iter),
                        tol=float(tol_per_frame) * len(X), random_state=int(seed), min_covar=1e-3,
                        init_params="stmc", params="stmc")
    model.fit(X, np.asarray(lengths, dtype=int))
    return model


class TorchGaussianHMM:
    """Full-covariance Gaussian HMM, EM (Baum-Welch) and Viterbi in log space with torch.

    Faster than ``hmmlearn`` at ~10^6 frames because the forward / backward recursions run over
    many sequences at once. Sequences longer than ``window`` frames are cut into pieces (the
    transitions across a cut are not counted, a negligible loss for pieces of hundreds of frames).
    Statistics are accumulated over batches of ``batch_pieces`` pieces, so memory stays bounded.
    """

    def __init__(self, n_states: int, seed: int = 0, device: str = "cpu", window: int = 512,
                 batch_pieces: int = 256, reg: float = 1e-3, self_prob: float = 0.95, dtype=None):
        import torch

        self.torch = torch
        self.K = int(n_states)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype or (torch.float32 if self.device.type == "cuda" else torch.float64)
        self.window, self.batch_pieces, self.reg, self.self_prob = int(window), int(batch_pieces), float(reg), float(self_prob)
        self.history: list[float] = []

    # ---------------------------------------------------------------- data layout
    def _pieces(self, lengths):
        out, start = [], 0
        for n in np.asarray(lengths, dtype=int):
            for a in range(0, n, self.window):
                out.append((start + a, min(self.window, n - a)))
            start += n
        return out

    def _batches(self, X, lengths):
        torch = self.torch
        pieces = self._pieces(lengths)
        for i in range(0, len(pieces), self.batch_pieces):
            chunk = pieces[i:i + self.batch_pieces]
            W = max(n for _s, n in chunk)
            idx = np.zeros((len(chunk), W), dtype=np.int64)
            mask = np.zeros((len(chunk), W), dtype=bool)
            for r, (s, n) in enumerate(chunk):
                idx[r, :n] = np.arange(s, s + n)
                idx[r, n:] = s + n - 1
                mask[r, :n] = True
            xb = torch.as_tensor(X[idx], dtype=self.dtype, device=self.device)
            yield idx, torch.as_tensor(mask, device=self.device), xb

    # ---------------------------------------------------------------- parameters
    def _init(self, X):
        from sklearn.cluster import KMeans

        rng = np.random.default_rng(self.seed)
        sub = X[rng.choice(len(X), size=min(len(X), 200_000), replace=False)]
        km = KMeans(self.K, n_init=3, random_state=self.seed).fit(sub)
        d = X.shape[1]
        cov = np.empty((self.K, d, d))
        glob = np.cov(sub.T)
        for k in range(self.K):
            pts = sub[km.labels_ == k]
            cov[k] = np.cov(pts.T) if len(pts) > d + 1 else glob
        A = np.full((self.K, self.K), (1 - self.self_prob) / max(self.K - 1, 1))
        np.fill_diagonal(A, self.self_prob if self.K > 1 else 1.0)
        self._set(np.full(self.K, 1.0 / self.K), A, km.cluster_centers_, cov)

    def _set(self, pi, A, means, cov):
        torch = self.torch
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)
        d = means.shape[1]
        cov = np.asarray(cov) + self.reg * np.eye(d)[None]
        self.startprob_, self.transmat_, self.means_, self.covars_ = (np.asarray(pi, float), np.asarray(A, float),
                                                                      np.asarray(means, float), cov)
        self._log_pi = torch.log(t(pi).clamp_min(1e-300))
        self._log_A = torch.log(t(A).clamp_min(1e-300))
        self._mu = t(means)
        self._chol = torch.linalg.cholesky(t(cov))
        self._logdet = 2 * torch.log(torch.diagonal(self._chol, dim1=-2, dim2=-1)).sum(-1)

    def _log_b(self, xb):
        # (B, W, d) -> (B, W, K) Gaussian log densities
        torch = self.torch
        B, W, d = xb.shape
        x = xb.reshape(-1, d)
        out = torch.empty((x.shape[0], self.K), dtype=self.dtype, device=self.device)
        for k in range(self.K):
            z = torch.linalg.solve_triangular(self._chol[k], (x - self._mu[k]).T, upper=False)
            out[:, k] = -0.5 * (z * z).sum(0) - 0.5 * self._logdet[k] - 0.5 * d * np.log(2 * np.pi)
        return out.reshape(B, W, self.K)

    # ---------------------------------------------------------------- inference
    def _forward_backward(self, lb, mask):
        torch = self.torch
        B, W, K = lb.shape
        la = self._log_A
        alpha = torch.empty_like(lb)
        a = self._log_pi + lb[:, 0]
        alpha[:, 0] = a
        for t in range(1, W):
            new = torch.logsumexp(a[:, :, None] + la[None], dim=1) + lb[:, t]
            a = torch.where(mask[:, t, None], new, a)
            alpha[:, t] = a
        ll = torch.logsumexp(a, dim=1)
        beta = torch.zeros_like(lb)
        b = torch.zeros((B, K), dtype=lb.dtype, device=lb.device)
        for t in range(W - 2, -1, -1):
            new = torch.logsumexp(la[None] + (lb[:, t + 1] + b)[:, None, :], dim=2)
            b = torch.where(mask[:, t + 1, None], new, b)
            beta[:, t] = b
        return alpha, beta, ll

    def _e_step(self, X, lengths):
        torch = self.torch
        K, d = self.K, X.shape[1]
        s0 = torch.zeros(K, dtype=self.dtype, device=self.device)
        s1 = torch.zeros((K, d), dtype=self.dtype, device=self.device)
        s2 = torch.zeros((K, d, d), dtype=self.dtype, device=self.device)
        xi = torch.zeros((K, K), dtype=self.dtype, device=self.device)
        pi = torch.zeros(K, dtype=self.dtype, device=self.device)
        total = 0.0
        for _idx, mask, xb in self._batches(X, lengths):
            lb = self._log_b(xb)
            alpha, beta, ll = self._forward_backward(lb, mask)
            total += float(ll.sum())
            g = torch.exp(alpha + beta - ll[:, None, None]) * mask[..., None]
            pi += g[:, 0].sum(0)
            gf, xf = g.reshape(-1, K), xb.reshape(-1, d)
            s0 += gf.sum(0)
            s1 += gf.T @ xf
            s2 += torch.einsum("nk,ni,nj->kij", gf, xf, xf)
            for t in range(lb.shape[1] - 1):
                m = mask[:, t + 1]
                if not bool(m.any()):
                    break
                x = (alpha[m, t][:, :, None] + self._log_A[None] + (lb[m, t + 1] + beta[m, t + 1])[:, None, :]
                     - ll[m][:, None, None])
                xi += torch.exp(x).sum(0)
        return total, pi, xi, s0, s1, s2

    def fit(self, X, lengths, n_iter: int = 60, tol_per_frame: float = 1e-4, verbose: bool = False):
        X = np.asarray(X, dtype=float)
        self._init(X)
        prev = -np.inf
        for it in range(int(n_iter)):
            total, pi, xi, s0, s1, s2 = self._e_step(X, lengths)
            self.history.append(total)
            n0 = s0.clamp_min(1e-10)
            means = s1 / n0[:, None]
            cov = s2 / n0[:, None, None] - means[:, :, None] * means[:, None, :]
            A = xi / xi.sum(1, keepdim=True).clamp_min(1e-300)
            self._set((pi / pi.sum()).cpu().numpy(), A.cpu().numpy(), means.cpu().numpy(), cov.cpu().numpy())
            if verbose:
                print(f"  K={self.K} iter {it + 1}: log-likelihood {total:.1f}")
            if total - prev < tol_per_frame * len(X):
                break
            prev = total
        self.n_iter_ = len(self.history)
        return self

    def score(self, X, lengths) -> float:
        total = 0.0
        for _idx, mask, xb in self._batches(np.asarray(X, float), lengths):
            total += float(self._forward_backward(self._log_b(xb), mask)[2].sum())
        return total

    def predict(self, X, lengths) -> np.ndarray:
        """Viterbi path of every frame (pieces of ``window`` frames decoded separately)."""
        torch = self.torch
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X), dtype=np.int64)
        for idx, mask, xb in self._batches(X, lengths):
            lb = self._log_b(xb)
            B, W, K = lb.shape
            back = torch.zeros((B, W, K), dtype=torch.int64, device=self.device)
            dlt = self._log_pi + lb[:, 0]
            for t in range(1, W):
                cand = dlt[:, :, None] + self._log_A[None]
                best, arg = cand.max(dim=1)
                new = best + lb[:, t]
                m = mask[:, t, None]
                dlt = torch.where(m, new, dlt)
                back[:, t] = torch.where(m, arg, torch.arange(K, device=self.device)[None].expand(B, K))
            path = torch.empty((B, W), dtype=torch.int64, device=self.device)
            cur = dlt.argmax(dim=1)
            path[:, W - 1] = cur
            for t in range(W - 1, 0, -1):
                cur = back[torch.arange(B, device=self.device), t, cur]
                path[:, t - 1] = cur
            p = path.cpu().numpy()
            mk = mask.cpu().numpy()
            out[idx[mk]] = p[mk]
        return out

    @property
    def n_components(self):
        return self.K

    @property
    def n_features(self):
        return self.means_.shape[1]

    covariance_type = "full"


def n_parameters(model) -> int:
    k, d = model.n_components, model.n_features
    cov = {"full": k * d * (d + 1) // 2, "diag": k * d, "spherical": k, "tied": d * (d + 1) // 2}[model.covariance_type]
    return (k - 1) + k * (k - 1) + k * d + cov


def decode_frames(model, X: np.ndarray, ok: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Viterbi state of every frame (``-1`` where ``ok`` is False); each continuous run of ``ok`` frames
    inside one ``block`` is one sequence."""
    out = np.full(len(X), -1, dtype=np.int64)
    s, e = runs(ok, block)
    if len(s) == 0:
        return out
    idx = np.concatenate([np.arange(a, b) for a, b in zip(s, e)])
    out[idx] = model.predict(X[idx], lengths=(e - s))
    return out
