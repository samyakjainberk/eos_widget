"""
Datasets + seeded initialisation.

MIRRORS:  server.py  (load_cifar, load_sort, _load_cifar_raw, _find_cifar_dir, init_data_theta)
          ↔ index.html  (synthetic data built with mulberry32 in JS; CIFAR/sorting are GPU-backend only)

Synthetic data + the MLP init use mulberry32 (browser-identical trajectories). CIFAR loads the raw
pickle batches (no torchvision); sorting generates (x, sorted-x) regression pairs.
"""
import os
import math
import torch

from .rng import mulberry32, u32, gauss

_CIFAR_CACHE = {}


def _find_cifar_dir(cifar_dir=None):
    cands = [cifar_dir,
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cifar-10-batches-py"),
             "/nas/ucb/samsj/TestingPSTheory/eos_widget/data/cifar-10-batches-py",   # in-repo copy on shared NAS
             os.path.expanduser("~/.torch/cifar-10-batches-py"),
             os.path.expanduser("~/data/cifar-10-batches-py")]
    for c in cands:
        if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "data_batch_1")):
            return c
    return None


def _load_cifar_raw(cifar_dir=None):
    """All 5 train batches, per-channel normalised, (50000,3072) float + int labels. MIRRORS server._load_cifar_raw."""
    import pickle
    import numpy as np
    d = _find_cifar_dir(cifar_dir)
    if d is None:
        raise FileNotFoundError("CIFAR-10 raw batches not found — pass cifar_dir=<dir with data_batch_1..5>")
    if d in _CIFAR_CACHE:
        return _CIFAR_CACHE[d]
    xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(d, f"data_batch_{i}"), "rb") as f:
            b = pickle.load(f, encoding="bytes")
        xs.append(np.asarray(b[b"data"], dtype=np.float32))
        ys.extend(list(b[b"labels"]))
    X = np.concatenate(xs, 0).reshape(-1, 3, 32, 32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    X = ((X - mean) / std).reshape(-1, 3072)
    Y = np.asarray(ys, dtype=np.int64)
    _CIFAR_CACHE[d] = (X, Y)
    return X, Y


def load_cifar(n, seed, device, dtype, cifar_dir=None):
    """Seeded n-image subset; X (n,3072), Y (n,10) one-hot. MIRRORS server.load_cifar."""
    import numpy as np
    X, Y = _load_cifar_raw(cifar_dir)
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    idx = rng.permutation(X.shape[0])[:n]
    Xt = torch.tensor(X[idx], dtype=dtype, device=device)
    Yt = torch.zeros(n, 10, dtype=dtype, device=device)
    Yt[torch.arange(n), torch.tensor(Y[idx], dtype=torch.long, device=device)] = 1.0
    return Xt, Yt


def load_cifar2(n, seed, device, dtype, cifar_dir=None, ca=0, cb=1):
    """2-class SCALAR CIFAR-10: keep classes {ca,cb}, label ca→+1, cb→−1; X (n,3072), Y (n,1).
    Balanced seeded subset. MIRRORS server.load_cifar2 (so §16/§17 per-sample-Hessian run on real images)."""
    import numpy as np
    X, Y = _load_cifar_raw(cifar_dir)
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    ia = np.where(Y == int(ca))[0]; ib = np.where(Y == int(cb))[0]
    rng.shuffle(ia); rng.shuffle(ib)
    na = n // 2; nb = n - na
    idx = np.concatenate([ia[:na], ib[:nb]])
    lab = np.concatenate([np.ones(min(na, len(ia)), np.float32), -np.ones(min(nb, len(ib)), np.float32)])
    perm = rng.permutation(len(idx)); idx = idx[perm]; lab = lab[perm]
    Xt = torch.tensor(X[idx], dtype=dtype, device=device)
    Yt = torch.tensor(lab, dtype=dtype, device=device).reshape(-1, 1)
    return Xt, Yt


_MNIST_CACHE = {}


def _find_mnist_dir(cifar_dir=None):
    base = os.path.dirname(cifar_dir) if cifar_dir else None
    cands = [base and os.path.join(base, "mnist"),
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mnist"),
             "/nas/ucb/samsj/TestingPSTheory/eos_widget/data/mnist",
             os.path.expanduser("~/data/mnist"), os.path.expanduser("~/.torch/mnist")]
    for c in cands:
        if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "train-images-idx3-ubyte")):
            return c
    return None


def _load_mnist_raw(split="train", cifar_dir=None):
    """MNIST as (N,3072): 28×28 zero-padded to 32×32, replicated to 3 channels (drop-in for cifar-shaped
    MLP/CNN/VGG). Per-channel MNIST-normalised. + int labels. MIRRORS server._load_mnist_raw."""
    import numpy as np
    d = _find_mnist_dir(cifar_dir)
    if d is None:
        raise FileNotFoundError("MNIST raw idx files not found — expected data/mnist/{train,t10k}-images/labels-idx*-ubyte")
    key = d + "::" + split
    if key in _MNIST_CACHE:
        return _MNIST_CACHE[key]
    pre = "train" if split == "train" else "t10k"
    with open(os.path.join(d, f"{pre}-images-idx3-ubyte"), "rb") as f:
        f.read(16); img = np.frombuffer(f.read(), dtype=np.uint8)
    with open(os.path.join(d, f"{pre}-labels-idx1-ubyte"), "rb") as f:
        f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
    n = lab.shape[0]
    img = img.reshape(n, 28, 28).astype(np.float32) / 255.0
    pad = np.zeros((n, 32, 32), np.float32); pad[:, 2:30, 2:30] = img
    x = np.repeat(pad[:, None, :, :], 3, axis=1)
    x = (x - 0.1307) / 0.3081
    X = x.reshape(n, 3072); Y = lab.astype(np.int64)
    _MNIST_CACHE[key] = (X, Y)
    return X, Y


def load_mnist(n, seed, device, dtype, cifar_dir=None):
    """Seeded n-image MNIST subset; X (n,3072) [3×32×32 padded], Y (n,10) one-hot. MIRRORS server.load_mnist."""
    import numpy as np
    X, Y = _load_mnist_raw("train", cifar_dir)
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    idx = rng.permutation(X.shape[0])[:n]
    Xt = torch.tensor(X[idx], dtype=dtype, device=device)
    Yt = torch.zeros(n, 10, dtype=dtype, device=device)
    Yt[torch.arange(n), torch.tensor(Y[idx], dtype=torch.long, device=device)] = 1.0
    return Xt, Yt


def load_mnist2(n, seed, device, dtype, cifar_dir=None, ca=0, cb=1):
    """2-class SCALAR MNIST: classes {ca,cb} → ±1, X (n,3072), Y (n,1). MIRRORS server.load_mnist2."""
    import numpy as np
    X, Y = _load_mnist_raw("train", cifar_dir)
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    ia = np.where(Y == int(ca))[0]; ib = np.where(Y == int(cb))[0]
    rng.shuffle(ia); rng.shuffle(ib)
    na = n // 2; nb = n - na
    idx = np.concatenate([ia[:na], ib[:nb]])
    lab = np.concatenate([np.ones(min(na, len(ia)), np.float32), -np.ones(min(nb, len(ib)), np.float32)])
    perm = rng.permutation(len(idx)); idx = idx[perm]; lab = lab[perm]
    Xt = torch.tensor(X[idx], dtype=dtype, device=device)
    Yt = torch.tensor(lab, dtype=dtype, device=device).reshape(-1, 1)
    return Xt, Yt


def _load_cifar_test_raw(cifar_dir=None):
    """The 10000-image CIFAR-10 test_batch, per-channel normalised like the train batches."""
    import pickle
    import numpy as np
    d = _find_cifar_dir(cifar_dir)
    if d is None or not os.path.isfile(os.path.join(d, "test_batch")):
        return None
    key = d + "::test"
    if key in _CIFAR_CACHE:
        return _CIFAR_CACHE[key]
    with open(os.path.join(d, "test_batch"), "rb") as f:
        b = pickle.load(f, encoding="bytes")
    X = np.asarray(b[b"data"], dtype=np.float32).reshape(-1, 3, 32, 32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(1, 3, 1, 1)
    X = ((X - mean) / std).reshape(-1, 3072)
    Y = np.asarray(list(b[b"labels"]), dtype=np.int64)
    _CIFAR_CACHE[key] = (X, Y)
    return X, Y


def _two_class_test(raw, n_test, P, device, dtype):
    """Held-out 2-class SCALAR test set (classes {c2a,c2b} → ±1) from a (X_images, int_labels) test split."""
    import numpy as np
    Xa, Ya = raw
    ca, cb = int(P.get("c2a", 0)), int(P.get("c2b", 1))
    rng = np.random.RandomState((P["seed"] & 0x7FFFFFFF) ^ 0x5151)
    ia = np.where(Ya == ca)[0]; ib = np.where(Ya == cb)[0]
    rng.shuffle(ia); rng.shuffle(ib)
    na = n_test // 2; nb = n_test - na
    idx = np.concatenate([ia[:na], ib[:nb]])
    lab = np.concatenate([np.ones(min(na, len(ia)), np.float32), -np.ones(min(nb, len(ib)), np.float32)])
    perm = rng.permutation(len(idx)); idx = idx[perm]; lab = lab[perm]
    return (torch.tensor(Xa[idx], dtype=dtype, device=device),
            torch.tensor(lab, dtype=dtype, device=device).reshape(-1, 1))


def make_test_set(model, P, dataset, n_test, in_dim, out_dim, device, dtype, cifar_dir=None):
    """Held-out (X_te, Y_te) or None when a test split is ill-defined (fixed-input / sign-filtered
    synthetic). CIFAR uses the real test_batch; synthetic/sorting draw fresh points from a separate
    RNG stream so they never overlap the training set."""
    if n_test <= 0:
        return None
    import numpy as np
    if dataset == "owt":
        try:                                   # held-out sequences from the val split
            return load_owt(n_test, in_dim, (P["seed"] ^ 0x5151), device, split="val", owt_dir=cifar_dir)
        except Exception:
            return None
    if dataset == "cifar10":
        raw = _load_cifar_test_raw(cifar_dir)
        if raw is None:
            return None
        Xa, Ya = raw
        rng = np.random.RandomState((P["seed"] & 0x7FFFFFFF) ^ 0x5151)
        idx = rng.permutation(Xa.shape[0])[:n_test]
        Xt = torch.tensor(Xa[idx], dtype=dtype, device=device)
        Yt = torch.zeros(len(idx), 10, dtype=dtype, device=device)
        Yt[torch.arange(len(idx)), torch.tensor(Ya[idx], dtype=torch.long, device=device)] = 1.0
        return Xt, Yt
    if dataset == "cifar2":                              # held-out cifar test_batch, 2-class scalar ±1
        raw = _load_cifar_test_raw(cifar_dir)
        return _two_class_test(raw, n_test, P, device, dtype) if raw is not None else None
    if dataset == "mnist":                               # held-out MNIST t10k, 10-class one-hot
        try:
            Xa, Ya = _load_mnist_raw("test", cifar_dir)
        except FileNotFoundError:
            return None
        rng = np.random.RandomState((P["seed"] & 0x7FFFFFFF) ^ 0x5151)
        idx = rng.permutation(Xa.shape[0])[:n_test]
        Xt = torch.tensor(Xa[idx], dtype=dtype, device=device)
        Yt = torch.zeros(len(idx), 10, dtype=dtype, device=device)
        Yt[torch.arange(len(idx)), torch.tensor(Ya[idx], dtype=torch.long, device=device)] = 1.0
        return Xt, Yt
    if dataset == "mnist2":                              # held-out MNIST t10k, 2-class scalar ±1
        try:
            raw = _load_mnist_raw("test", cifar_dir)
        except FileNotFoundError:
            return None
        return _two_class_test(raw, n_test, P, device, dtype)
    if dataset == "chebyshev":
        return None                                      # fixed deterministic dataset — no held-out split
    trng = mulberry32(u32(P["seed"] * 7919 + 99991))     # separate stream ⇒ disjoint from train
    tgt, in_std = float(P["tgt"]), float(P["inputstd"])
    if dataset == "sorting":
        return load_sort(n_test, in_dim, trng, device, dtype)
    if dataset == "const":                                # held-out: fresh iid Gaussian X, same constant target |tgt| (+ cvar noise)
        if P.get("ssign", "off") != "off":
            return None                                    # sign-forced const ⇒ f₀-dependent targets, no clean held-out set (matches synthetic)
        cstd = max(0.0, float(P.get("cvar", 0.0))) ** 0.5
        Xl = [[in_std * gauss(trng) for _ in range(in_dim)] for _ in range(n_test)]
        Yl = [[abs(tgt) + cstd * gauss(trng) for _ in range(out_dim)] for _ in range(n_test)]
        return (torch.tensor(Xl, dtype=dtype, device=device),
                torch.tensor(Yl, dtype=dtype, device=device))
    # synthetic: only well-defined for iid Gaussian inputs with off sign-mode
    if P["fixedx"] == "1" or P.get("ssign", "off") != "off":
        return None
    Xl = [[in_std * gauss(trng) for _ in range(in_dim)] for _ in range(n_test)]
    Yl = [[tgt * gauss(trng) for _ in range(out_dim)] for _ in range(n_test)]
    return (torch.tensor(Xl, dtype=dtype, device=device),
            torch.tensor(Yl, dtype=dtype, device=device))


def load_sort(n, L, drng, device, dtype):
    """Sorting task: X random length-L, Y = ascending-sorted X (MSE, oc=L). MIRRORS server.load_sort."""
    Xl = [[gauss(drng) for _ in range(L)] for _ in range(n)]
    X = torch.tensor(Xl, dtype=dtype, device=device)
    Y, _ = torch.sort(X, dim=1)
    return X, Y


def load_chebyshev(n, k, device, dtype):
    """Chebyshev regression (Cohen et al. EoS toy task): n points evenly spaced on [-1,1] (input, n×1),
    labeled by the Chebyshev polynomial T_k of degree k (output, n×1). MSE. T_k via the recurrence
    T₀=1, T₁=x, Tₘ=2x·Tₘ₋₁−Tₘ₋₂. MIRRORS server.load_chebyshev."""
    x = torch.linspace(-1.0, 1.0, max(int(n), 1), dtype=dtype, device=device)
    if int(k) <= 0:
        y = torch.ones_like(x)
    else:
        tm2, tm1 = torch.ones_like(x), x.clone()
        for _ in range(2, int(k) + 1):
            tm2, tm1 = tm1, 2 * x * tm1 - tm2
        y = tm1
    return x.unsqueeze(1), y.unsqueeze(1)


_OWT_CACHE = {}


def _find_owt_dir(owt_dir=None):
    cands = [owt_dir,
             os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "owt"),
             os.path.expanduser("~/data/owt")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "train.bin")):
            return c
    return None


def _owt_tokens(split, owt_dir=None):
    """Memory-mapped uint16 token array for the train/val split. MIRRORS server._owt_tokens."""
    import numpy as np
    d = _find_owt_dir(owt_dir)
    if d is None:
        raise FileNotFoundError("OpenWebText tokens not found — run `python -m eos_lab.owt_prepare` "
                                "to build data/owt/{train,val}.bin (or pass owt_dir=...).")
    key = (d, split)
    if key not in _OWT_CACHE:
        _OWT_CACHE[key] = np.memmap(os.path.join(d, f"{split}.bin"), dtype=np.uint16, mode="r")
    return _OWT_CACHE[key]


def load_owt(n, block, seed, device, split="train", owt_dir=None):
    """`n` length-`block` token sequences sampled at random offsets from the OWT corpus.
    Returns (X, Y) int64 (n, block): X = tokens[off:off+block], Y = next tokens[off+1:off+block+1].
    MIRRORS server.load_owt. (X/Y are token ids — independent of the model's float dtype.)"""
    import numpy as np
    tok = _owt_tokens(split, owt_dir)
    hi = len(tok) - block - 1
    if hi <= 0:
        raise ValueError(f"OWT {split}.bin has {len(tok)} tokens; need > block+1 = {block + 1}.")
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    offs = rng.randint(0, hi, size=n)
    X = np.stack([np.asarray(tok[o:o + block], dtype=np.int64) for o in offs])
    Y = np.stack([np.asarray(tok[o + 1:o + 1 + block], dtype=np.int64) for o in offs])
    return (torch.tensor(X, dtype=torch.long, device=device),
            torch.tensor(Y, dtype=torch.long, device=device))


def init_data_theta(model, P, dataset, N, in_dim, out_dim, device, dtype, cifar_dir=None):
    """Return (theta0, X, Y, pos_rows, neg_rows). MIRRORS server.init_data_theta.
    pos_rows/neg_rows are the flat residual-row indices (i·out_dim+c) split by the sign of the
    summed initial residual r=y−f(x,θ₀) — they drive the §4d per-group projection panel."""
    th = model.init_theta(P["seed"] + 1, P["init"])
    drng = mulberry32(u32(P["seed"] * 7919 + 1))
    ssign = P.get("ssign", "off")
    tgt = float(P["tgt"])
    in_std = float(P["inputstd"])
    fixedx = P["fixedx"] == "1"

    if dataset == "owt":
        # `in_dim` = block size; X/Y are int token ids. CE has no residual geometry, so §4d is unused
        # → pos_rows/neg_rows are None (and the residual-sign machinery below is skipped entirely).
        X, Y = load_owt(N, in_dim, P["seed"], device, owt_dir=cifar_dir)
        return th, X, Y, None, None

    if dataset == "cifar10":
        X, Y = load_cifar(N, P["seed"], device, dtype, cifar_dir)
    elif dataset == "cifar2":
        X, Y = load_cifar2(N, P["seed"], device, dtype, cifar_dir, P.get("c2a", 0), P.get("c2b", 1))
    elif dataset == "mnist":
        X, Y = load_mnist(N, P["seed"], device, dtype, cifar_dir)
    elif dataset == "mnist2":
        X, Y = load_mnist2(N, P["seed"], device, dtype, cifar_dir, P.get("c2a", 0), P.get("c2b", 1))
    elif dataset == "sorting":
        X, Y = load_sort(N, in_dim, drng, device, dtype)
    elif dataset == "chebyshev":
        X, Y = load_chebyshev(N, P.get("degree", 3), device, dtype)
    elif dataset == "const":
        # iid Gaussian inputs; target = CONSTANT POSITIVE |tgt| + Gaussian noise of variance cvar (0 ⇒ exact
        # constant ⇒ uniform residuals; cvar>0 decorrelates the residuals). MIRRORS server.
        cstd = max(0.0, float(P.get("cvar", 0.0))) ** 0.5
        Xl = [[in_std * gauss(drng) for _ in range(in_dim)] for _ in range(N)]
        X = torch.tensor(Xl, dtype=dtype, device=device)
        Ymag = torch.tensor([[abs(tgt) + cstd * gauss(drng) for _ in range(out_dim)] for _ in range(N)],
                            dtype=dtype, device=device)                   # |tgt| (+cvar noise); drawn for EVERY ssign (RNG parity)
        if ssign in ("pos", "neg"):                                       # FORCE all initial residual signs: r=y−f=s·max(|tgt|+noise,floor) ⇒ sign s
            s = 1.0 if ssign == "pos" else -1.0
            floor = 0.25 * max(abs(tgt), 1e-6)
            Y = model.forward(th, X) + s * Ymag.clamp(min=floor)         # uniform |tgt| residual with the chosen sign (cvar still spreads it)
        else:
            Y = Ymag                                                      # off: constant positive target |tgt| (+noise); residuals naturally +
    elif fixedx:
        X = torch.ones(N, in_dim, dtype=dtype, device=device)
        if ssign == "off":
            Y = torch.full((N, out_dim), tgt, dtype=dtype, device=device)
        else:
            s = 1.0 if ssign == "pos" else -1.0
            floor = 0.25 * max(abs(tgt), 1e-6)
            Fout = model.forward(th, X)
            Y = Fout + s * torch.clamp(torch.full_like(Fout, abs(tgt)), min=floor)
    elif ssign == "off":
        Xl = [[in_std * gauss(drng) for _ in range(in_dim)] for _ in range(N)]
        Yl = [[tgt * gauss(drng) for _ in range(out_dim)] for _ in range(N)]
        X = torch.tensor(Xl, dtype=dtype, device=device)
        Y = torch.tensor(Yl, dtype=dtype, device=device)
    else:
        # keep the N samples whose initial residual r=y−f(x,θ₀) is sign-definite with largest |r|
        s = 1.0 if ssign == "pos" else -1.0
        floor = 0.25 * max(abs(tgt), 1e-6)
        K = max(64 * N, 3000)
        Xc = torch.tensor([[in_std * gauss(drng) for _ in range(in_dim)] for _ in range(K)],
                          dtype=dtype, device=device)
        Yc = torch.tensor([[tgt * gauss(drng) for _ in range(out_dim)] for _ in range(K)],
                          dtype=dtype, device=device)
        Fc = model.forward(th, Xc)
        Rc = Yc - Fc
        ok = (Rc * s > 0).all(dim=1)
        order = torch.argsort(Rc.abs().amin(dim=1), descending=True).tolist()
        chosen = [k for k in order if bool(ok[k])]
        for k in order:
            if len(chosen) >= N:
                break
            if not bool(ok[k]):
                chosen.append(k)
        chosen = chosen[:N]
        X = torch.empty(N, in_dim, dtype=dtype, device=device)
        Y = torch.empty(N, out_dim, dtype=dtype, device=device)
        for i, k in enumerate(chosen):
            X[i] = Xc[k]
            f = Fc[k]
            Y[i] = f + s * torch.clamp((Yc[k] - f).abs(), min=floor)
    # Fixed-target datasets (cifar10/sorting/chebyshev) load Y directly and so skip the residual-sign
    # construction above — force the requested initial residual sign here by overriding Y per sample
    # (keep the dataset's inputs X and the residual's natural magnitude; only its sign is pinned).
    if dataset in ("cifar10", "cifar2", "mnist", "mnist2", "sorting", "chebyshev") and ssign in ("pos", "neg"):
        s = 1.0 if ssign == "pos" else -1.0
        floor = 0.25 * max(abs(tgt), 1e-6)
        f0 = model.forward(th, X)
        Y = f0 + s * torch.clamp((Y - f0).abs(), min=floor)

    # §4d: fix the two sample groups by the sign of the summed initial residual r=y−f(x,θ₀)
    ssum0 = (Y - model.forward(th, X)).sum(dim=1)
    pos_rows = torch.tensor([i * out_dim + c for i in range(N) if float(ssum0[i]) >= 0
                             for c in range(out_dim)], dtype=torch.long, device=device)
    neg_rows = torch.tensor([i * out_dim + c for i in range(N) if float(ssum0[i]) < 0
                             for c in range(out_dim)], dtype=torch.long, device=device)
    return th, X, Y, pos_rows, neg_rows
