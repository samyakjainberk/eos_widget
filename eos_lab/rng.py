"""
Deterministic RNG — byte-for-byte identical to the browser widget.

MIRRORS:  server.py  (u32/i32/imul/xor32/or32/mulberry32/gauss)  ↔  index.html  (mulberry32, gauss)

Why this exists: the synthetic-data presets and the MLP weight init use `mulberry32` (a tiny
32-bit PRNG) and a Box–Muller `gauss`. Reproducing them exactly here means a given `seed`
produces the SAME gradient-descent trajectory as the in-browser widget and as `server.py`.
CIFAR / sorting / the conv & transformer inits use torch's RNG instead (see models.py / data.py).
"""
import math

MASK32 = 0xFFFFFFFF


def u32(x):
    """x as an unsigned 32-bit int."""
    return x & MASK32


def i32(x):
    """x as a signed 32-bit int (JS `| 0` semantics)."""
    x &= MASK32
    return x - 0x100000000 if x >= 0x80000000 else x


def imul(a, b):
    """JS Math.imul: 32-bit integer multiply with wraparound, signed result."""
    return i32((a & MASK32) * (b & MASK32))


def xor32(a, b):
    return i32(a ^ b)


def or32(a, b):
    return i32(a | b)


def mulberry32(seed):
    """Return a callable rnd() -> float in [0,1). Identical stream to the JS mulberry32."""
    st = {"a": i32(seed)}

    def rnd():
        a = i32(st["a"] + 0x6D2B79F5)
        st["a"] = a
        t = imul(xor32(a, u32(a) >> 15), or32(1, a))
        t = xor32(i32(t + imul(xor32(t, u32(t) >> 7), or32(61, t))), t)
        return u32(xor32(t, u32(t) >> 14)) / 4294967296.0

    return rnd


def gauss(rng):
    """One standard normal via Box–Muller, drawing from a mulberry32 stream `rng`."""
    u = 0.0
    v = 0.0
    while u == 0:
        u = rng()
    while v == 0:
        v = rng()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)
