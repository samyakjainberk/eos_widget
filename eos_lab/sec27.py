"""§27 — sliding-window 3D subspace projection of six ∇θ gradient-vectors, coloured by iteration.

Six per-step p-dimensional gradient vectors (exact autograd, torch.func — same primitives as §25):
  f  : ∇‖f‖²            (φ6 — model output)
  r  : ∇‖r‖²            (φ1 — residual, r = Y−f)
  J  : ∇‖J‖²_F          (φ2 — Jacobian Frobenius, = ∇ tr NTK)
  g  : ∇‖∇L‖²           (φ3 — loss-gradient norm)
  rd : ∇‖ṙ‖²  = −2 Jᵀ(r_t − r_{t−1})              (DISCRETE step-difference; 0 at t=0)
  Jd : ∇‖J̇‖²_F = ∇‖J‖²_F − 2 ∇⟨J, J_{t−1}⟩_F      (DISCRETE step-difference; 0 at t=0)
where ⟨J, J_{t−1}⟩_F = Σ_ij J_ij (J_{t−1})_ij is a frozen contraction against the previous step's
Jacobian, so ∇⟨J,J_{t−1}⟩_F is one double-backward (jacrev inside grad) at the current θ.

For every iteration t we stack the window [t−50, t+50] (whatever is available — clipped at both ends)
of a metric's vectors into A (p × W) and read the top-3 LEFT singular directions; the projection of
the t-th (center) column onto them reduces, via the Gram G = AᵀA with eigenpairs (σ_i², v_i), to
    coord_i = σ_i · v_i[center]        (no p-dimensional SVD needed).
Panels 1-2 use a per-metric subspace; panels 3-4 use a COMMON subspace = top-3 of the horizontally
stacked [A_f | A_r | A_J | A_g | A_rd | A_Jd] (p × 6W), each metric's center column projected onto it.

Because the +50 half-window needs FUTURE iterations, §27 is emitted with a 50-STEP LAG: a rolling
buffer holds the last 101 vector-sets and at GD step T we finalize iteration t = T−50 (its +50
neighbours are now buffered). Head iterations (t<50) get left-clipped windows; the trailing 50 are
flushed at end-of-run with right-clipped windows (Sec27State.flush). Sign continuity: each window's
singular directions are sign-aligned to the previous finalized iteration's over the shared columns
(the windows overlap in 100/101 columns, so the subspace is stable) — this keeps the coloured curves
from flipping sign between adjacent points.

The browser only RENDERS g27 (12 scatter3d curves); it does not recompute (same as §26). Parity is
server ↔ eos_lab: both build the same Gram from the same vectors and take torch.linalg.eigh, so the
coords match to eigh tolerance (looser than the exact-autograd sections where singular values are
near-degenerate, since the eigenvector basis is then ambiguous — the sign-continuity mitigates it).
"""
import torch

HALF = 50                 # window half-width  ⇒ [t−HALF, t+HALF]
CAP = 2 * HALF + 1        # 101 rolling buffer entries
TOP = 3                   # subspace dimension (3D plots)
METRICS = ("f", "r", "J", "g", "rd", "Jd")   # fixed order (panels: 1={f,r,J} 2={g,rd,Jd}, 3/4 same via common subspace)


def sec27_vectors(model, loss, th, X, Y, N, outD, Jc, rr, prev):
    """The six ∇θ metric vectors at the current step (each detached, shape (p,)).

    prev : None on the first step, else {"r": r_{t−1} (M,), "J": J_{t−1} (M,p)} from the last step.
    Returns (vecs: dict metric→(p,) tensor, cur: {"r","J"} to carry forward as the next `prev`)."""
    import torch.func as _tf
    M = N * outD
    Jm = Jc[:M]                                       # (M,p) effective-sample Jacobian
    rm = rr[:M]                                       # (M,) residual  (r = Y−f for MSE)
    Yf = Y.reshape(-1)
    ff = lambda q: model.forward(q, X).reshape(-1)    # f(θ) flat (M,)
    Lval = lambda q: loss.value(model.forward(q, X), Y, N)

    def phi_f(q):
        f = ff(q); return (f * f).sum()               # ‖f‖²
    def phi_r(q):
        d = Yf - ff(q); return (d * d).sum()          # ‖r‖²
    def phi_J(q):
        J = _tf.jacrev(ff)(q); return (J * J).sum()   # ‖J‖²_F
    def phi_g(q):
        g = _tf.grad(Lval)(q); return (g * g).sum()   # ‖∇L‖²

    m_f = _tf.grad(phi_f)(th).detach()
    m_r = _tf.grad(phi_r)(th).detach()
    m_J = _tf.grad(phi_J)(th).detach()
    m_g = _tf.grad(phi_g)(th).detach()
    if prev is None:                                   # ṙ, J̇ undefined at t=0 (no previous step) ⇒ zero vectors
        z = torch.zeros_like(m_f)
        m_rd = z
        m_Jd = z.clone()
    else:
        dr = rm - prev["r"]                            # Δr = r_t − r_{t−1}   (M,)
        m_rd = (-2.0 * (Jm.t() @ dr)).detach()         # ∇‖ṙ‖² = −2 Jᵀ Δr
        C = prev["J"]                                  # J_{t−1}  (M,p), frozen
        cross = _tf.grad(lambda q: (_tf.jacrev(ff)(q) * C).sum())(th).detach()   # ∇⟨J,J_{t−1}⟩_F
        m_Jd = (m_J - 2.0 * cross).detach()            # ∇‖J̇‖²_F = ∇‖J‖²_F − 2∇⟨J,J_{t−1}⟩_F
    vecs = {"f": m_f, "r": m_r, "J": m_J, "g": m_g, "rd": m_rd, "Jd": m_Jd}
    cur = {"r": rm.detach().clone(), "J": Jm.detach().clone()}
    return vecs, cur


def _top_eig(gram, top):
    """Top-`top` eigenpairs (descending) of a symmetric PSD Gram; pads to `top` if it is smaller."""
    n = gram.shape[0]
    k = min(top, n)
    evals, evecs = torch.linalg.eigh(gram)             # ascending
    order = torch.argsort(evals, descending=True)[:k]
    lam = evals[order].clamp_min(0.0)
    V = evecs[:, order]                                # (n,k)
    sig = lam.sqrt()
    if k < top:                                        # very short windows (W<3): pad missing modes with 0
        sig = torch.cat([sig, torch.zeros(top - k, dtype=gram.dtype, device=gram.device)])
        V = torch.cat([V, torch.zeros(n, top - k, dtype=gram.dtype, device=gram.device)], dim=1)
    return sig, V


def _sign_align(prev, keys, V):
    """Flip each mode's sign in place so V matches `prev` over shared column-keys (heavy window
    overlap ⇒ stable), returning a boolean flip mask and the state {"keys","V"} to store for next t.
    `keys` labels V's rows (metric-iter identity); `prev` is the previous finalized iteration's state."""
    top = V.shape[1]
    flip = [False] * top
    if prev is not None:
        pidx = {kk: i for i, kk in enumerate(prev["keys"])}
        cur = [(i, pidx[kk]) for i, kk in enumerate(keys) if kk in pidx]
        if cur:
            ci = torch.tensor([i for i, _ in cur], device=V.device)
            pi = torch.tensor([j for _, j in cur], device=V.device)
            for j in range(top):
                if float((V[ci, j] * prev["V"][pi, j]).sum()) < 0:
                    V[:, j] = -V[:, j]
                    flip[j] = True
    return flip, {"keys": list(keys), "V": V.detach().clone()}


class Sec27State:
    """Rolling-buffer state for the 50-lag streaming projection. One instance per run (per backend)."""

    def __init__(self, half=HALF, top=TOP):
        self.half = half
        self.top = top
        self.cap = 2 * half + 1
        self.buf = []            # last ≤cap ticks {t, vecs}, newest last
        self.n_seen = 0          # total ticks pushed (POSITION counter — robust to eigevery>1 & resume: t need not be 0,1,2,…)
        self.n_final = 0         # ticks finalized so far
        self.sign = {}           # key → {"keys","V"}: per-metric ("f".."Jd") + common ("c") sign references

    def push_step(self, t, vecs):
        """Append this tick's six vectors; return the ticks that just became complete (their +HALF forward
        neighbour is now buffered). Indexed by POSITION (not the t value), so it is correct for non-consecutive
        t (eigevery>1) and for a resumed run whose first t>0."""
        self.buf.append({"t": t, "vecs": vecs})
        if len(self.buf) > self.cap:
            self.buf.pop(0)
        self.n_seen += 1
        pts = []
        while self.n_seen - self.n_final > self.half:    # tick at position n_final now has its +HALF future
            pts.append(self._finalize(self.n_final))
            self.n_final += 1
        return pts

    def flush(self):
        """End-of-run: finalize the trailing ticks (right-clipped windows) still in the buffer."""
        pts = []
        while self.n_final < self.n_seen:
            pts.append(self._finalize(self.n_final))
            self.n_final += 1
        return pts

    def _finalize(self, pos):
        buf_start = self.n_seen - len(self.buf)          # absolute position of buf[0]
        lo = max(0, pos - self.half)
        hi = min(self.n_seen - 1, pos + self.half)       # right-clipped for the tail; = pos+HALF during streaming
        win = self.buf[lo - buf_start: hi - buf_start + 1]
        iters = [e["t"] for e in win]
        center = pos - lo                                # index of the finalized tick within its window
        W = len(win)
        perM, common = {}, {}
        # ---- panels 1-2: per-metric subspace ----
        for m in METRICS:
            cols = torch.stack([e["vecs"][m] for e in win], dim=1)     # (p,W)
            sig, V = _top_eig(cols.t() @ cols, self.top)               # eig of the W×W Gram
            flip, self.sign[m] = _sign_align(self.sign.get(m), iters, V)
            coords = sig * V[center, :]                                # coord_i = σ_i · v_i[center]
            perM[m] = [float(coords[j]) for j in range(self.top)]
        # ---- panels 3-4: common subspace (all six metrics stacked side by side) ----
        big = torch.cat([torch.stack([e["vecs"][m] for e in win], dim=1) for m in METRICS], dim=1)  # (p,6W)
        keys_c = [(m, it) for m in METRICS for it in iters]            # 6W column identities
        sig_c, V_c = _top_eig(big.t() @ big, self.top)
        _, self.sign["c"] = _sign_align(self.sign.get("c"), keys_c, V_c)   # flips V_c in place
        for mi, m in enumerate(METRICS):
            col = mi * W + center                                      # center column of metric m in the stack
            common[m] = [float(sig_c[j] * V_c[col, j]) for j in range(self.top)]
        return {"t": iters[center], "perM": perM, "common": common}
