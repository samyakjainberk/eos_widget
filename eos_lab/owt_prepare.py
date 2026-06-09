"""
Prepare a *small* GPT-2-BPE OpenWebText shard for the `owt` dataset option.

Writes two memory-mappable uint16 token arrays (nanoGPT / data_owt_prepare.py convention):
    <out_dir>/train.bin
    <out_dir>/val.bin
plus a meta.json recording the vocab size and token counts.

Unlike the full OWT prep (~8 h / ~9 B tokens), this streams just enough documents to reach a small
target (default ~20 M tokens) — plenty for the EoS tool, whose runs use at most a few thousand
length-`block` sequences. Falls back to a deterministic synthetic-token stub if HuggingFace is
unreachable, so the `owt` option always has *something* to load.

    python -m eos_lab.owt_prepare --out-dir eos_widget/data/owt --target-tokens 20000000
"""
import argparse
import json
import os

import numpy as np


def _tiktoken_enc():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def _stream_tokens(enc, target_tokens, eot):
    """Yield token ids from streamed OWT docs until `target_tokens` reached. Returns a flat list."""
    from datasets import load_dataset
    # streaming avoids the ~40 GB full download; try the canonical corpus then a tiny mirror.
    last = None
    for name, kw in [("Skylion007/openwebtext", dict(split="train", streaming=True,
                                                     trust_remote_code=True)),
                     ("stas/openwebtext-10k", dict(split="train", trust_remote_code=True))]:
        try:
            ds = load_dataset(name, **kw)
        except Exception as e:           # network / dataset-script issues — try the next source
            last = e
            continue
        toks = []
        for ex in ds:
            toks.extend(enc.encode_ordinary(ex["text"]))
            toks.append(eot)
            if len(toks) >= target_tokens:
                break
        if toks:
            return toks, name
    raise RuntimeError(f"could not stream any OWT source (last error: {last})")


def _stub_tokens(target_tokens, vocab, eot, seed=0):
    """Deterministic synthetic 'corpus' so the owt option works with no network. Repeating, mildly
    structured token stream (not language, but exercises the full LM-GPT + CE pipeline)."""
    rng = np.random.RandomState(seed)
    # a small recurring vocabulary of 'words' (short token runs) stitched into documents
    words = [list(rng.randint(0, vocab, size=int(rng.randint(1, 6)))) for _ in range(2000)]
    toks = []
    while len(toks) < target_tokens:
        for _ in range(int(rng.randint(20, 60))):
            toks.extend(words[rng.randint(len(words))])
        toks.append(eot)
    return toks[:target_tokens]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare a small GPT-2-BPE OpenWebText shard.")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "owt"))
    ap.add_argument("--target-tokens", type=int, default=20_000_000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--stub", action="store_true", help="skip the download, write a synthetic stub")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    vocab, eot, source = 50257, 50256, "stub"        # gpt2: 50257 tokens, <|endoftext|> = 50256
    if a.stub:
        toks = _stub_tokens(a.target_tokens, vocab, eot)
    else:
        try:
            enc = _tiktoken_enc()
            toks, source = _stream_tokens(enc, a.target_tokens, eot)
        except Exception as e:
            print(f"  download failed ({e}); writing synthetic stub instead.")
            toks = _stub_tokens(a.target_tokens, vocab, eot)

    arr = np.asarray(toks, dtype=np.uint16)
    n_val = int(len(arr) * a.val_frac)
    splits = {"train": arr[:-n_val] if n_val else arr, "val": arr[-n_val:] if n_val else arr[:1]}
    for name, a_ in splits.items():
        path = os.path.join(a.out_dir, f"{name}.bin")
        m = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(a_),))
        m[:] = a_
        m.flush()
        print(f"  wrote {path}  ({len(a_):,} tokens)")
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump({"vocab_size": vocab, "eot": eot, "source": source,
                   "train_tokens": int(len(splits["train"])), "val_tokens": int(len(splits["val"]))}, f, indent=2)
    print(f"done: source={source}  total={len(arr):,} tokens  -> {a.out_dir}")


if __name__ == "__main__":
    main()
