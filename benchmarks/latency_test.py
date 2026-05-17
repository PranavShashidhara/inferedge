"""
benchmarks/latency_test.py
--------------------------
Measures time-per-token and end-to-end latency across sequence lengths
and batch sizes.  Produces a DataFrame and matplotlib heatmaps.

OOM-safe: skips (seq_len, batch_size) combinations that exceed GPU memory
and continues the sweep rather than crashing.
"""

from __future__ import annotations

import gc
import time
import random
import string
from typing import List, Optional

import torch
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------

def run_latency_benchmark(
    decoder,
    seq_lengths:    List[int] = (128, 256, 512, 1024, 2048),
    batch_sizes:    List[int] = (1, 2, 4, 8),
    num_trials:     int       = 5,
    warmup_trials:  int       = 2,
    max_new_tokens: int       = 32,
    skip_on_oom:    bool      = True,
) -> pd.DataFrame:
    """
    Sweep over (seq_len, batch_size) pairs and measure:
      - prefill latency  (ms)
      - mean per-token decode latency  (ms/token)
      - total end-to-end latency  (ms)

    Parameters
    ----------
    decoder        : runtime.decoder.Decoder instance
    seq_lengths    : input prompt lengths to test
    batch_sizes    : batch sizes to test
    num_trials     : timed repetitions per configuration
    warmup_trials  : un-timed warm-up runs (fills GPU caches)
    max_new_tokens : tokens to generate per trial
    skip_on_oom    : if True, catches OOM errors and continues the sweep
                     instead of crashing; the skipped row is recorded as NaN

    Returns
    -------
    pandas DataFrame with one row per (seq_len, batch_size) configuration.
    Skipped configurations have NaN for metric columns.
    """
    records = []

    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            label = f"seq_len={seq_len:5d}  batch_size={batch_size}"
            print(f"  {label} ...", end=" ", flush=True)

            # Build prompts outside the try block so we can report failures
            try:
                prompts = [
                    _make_prompt(seq_len, decoder.tokenizer)
                    for _ in range(batch_size)
                ]
            except Exception as e:
                print(f"prompt build failed: {e}")
                records.append(_nan_record(seq_len, batch_size))
                continue

            # --- Warm-up -------------------------------------------------
            try:
                for _ in range(warmup_trials):
                    decoder.generate_batch(prompts, max_new_tokens=4)
                torch.cuda.synchronize()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if skip_on_oom and _is_oom(e):
                    print("OOM during warm-up — skipping")
                    _recover_gpu()
                    records.append(_nan_record(seq_len, batch_size))
                    continue
                raise

            # --- Timed runs ----------------------------------------------
            e2e_times:     List[float] = []
            per_tok_times: List[float] = []
            oom_hit = False

            for trial in range(num_trials):
                try:
                    t0 = time.perf_counter()
                    decoder.generate_batch(prompts, max_new_tokens=max_new_tokens)
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                    e2e_times.append(elapsed_ms)
                    per_tok_times.append(elapsed_ms / (batch_size * max_new_tokens))

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if skip_on_oom and _is_oom(e):
                        print(f"OOM on trial {trial+1} — skipping config")
                        _recover_gpu()
                        oom_hit = True
                        break
                    raise

            if oom_hit:
                records.append(_nan_record(seq_len, batch_size))
                continue

            # --- Aggregate -----------------------------------------------
            mean_e2e     = sum(e2e_times)     / len(e2e_times)
            mean_per_tok = sum(per_tok_times) / len(per_tok_times)
            p90_per_tok  = sorted(per_tok_times)[max(0, int(0.9 * len(per_tok_times)) - 1)]

            # Extract prefill latency from kernel stats
            stats = decoder.get_last_kernel_stats()
            prefill_ms = next(
                (s["latency_ms"] for s in stats if s["phase"] == "prefill"), 0.0
            )

            print(f"e2e={mean_e2e:.1f}ms  tok={mean_per_tok:.2f}ms")

            records.append({
                "seq_len":        seq_len,
                "batch_size":     batch_size,
                "prefill_ms":     prefill_ms,
                "e2e_ms":         mean_e2e,
                "per_tok_ms":     mean_per_tok,
                "p90_per_tok_ms": p90_per_tok,
                "tokens_per_sec": 1000.0 / mean_per_tok,
                "oom":            False,
            })

    df = pd.DataFrame(records)
    _save_csv(df, "results/csv_logs/latency_results.csv")

    n_skipped = df["oom"].sum() if "oom" in df.columns else 0
    if n_skipped:
        print(f"\n  ⚠  {int(n_skipped)} configuration(s) skipped due to OOM.")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_latency_results(df: pd.DataFrame) -> None:
    """
    Render three plots:
      1. Per-token latency heatmap  (seq_len × batch_size)
      2. Prefill latency vs seq_len
      3. Tokens/sec vs batch_size

    OOM-skipped cells are shown as grey in the heatmap.
    """
    # Only use rows that completed successfully
    valid = df[~df["oom"]].copy() if "oom" in df.columns else df.copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("LLM Inference Latency — A100", fontsize=14, fontweight="bold")

    # ---- 1. Heatmap: per-token latency ----------------------------------
    all_seqs   = sorted(df["seq_len"].unique())
    all_batches = sorted(df["batch_size"].unique())
    heat = pd.DataFrame(index=all_seqs, columns=all_batches, dtype=float)
    for _, row in df.iterrows():
        heat.loc[row["seq_len"], row["batch_size"]] = (
            float("nan") if row.get("oom", False) else row["per_tok_ms"]
        )

    ax = axes[0]
    import numpy as np
    arr = heat.values.astype(float)
    # Mask NaNs for colour scaling
    vmin = float(np.nanmin(arr)) if not np.all(np.isnan(arr)) else 0
    vmax = float(np.nanmax(arr)) if not np.all(np.isnan(arr)) else 1

    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(all_batches)))
    ax.set_xticklabels([f"B={b}" for b in all_batches])
    ax.set_yticks(range(len(all_seqs)))
    ax.set_yticklabels(all_seqs)
    ax.set_title("Per-Token Latency (ms/token)")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Sequence length")
    plt.colorbar(im, ax=ax)

    for i, sl in enumerate(all_seqs):
        for j, bs in enumerate(all_batches):
            val = heat.loc[sl, bs]
            if pd.isna(val):
                ax.text(j, i, "OOM", ha="center", va="center",
                        fontsize=8, color="grey", fontweight="bold")
            else:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    # ---- 2. Prefill latency vs seq_len ----------------------------------
    ax2 = axes[1]
    for bs in sorted(valid["batch_size"].unique()):
        sub = valid[valid["batch_size"] == bs].sort_values("seq_len")
        ax2.plot(sub["seq_len"], sub["prefill_ms"], marker="o", label=f"B={bs}")
    ax2.set_title("Prefill Latency vs Sequence Length")
    ax2.set_xlabel("Sequence length (tokens)")
    ax2.set_ylabel("Prefill latency (ms)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---- 3. Tokens/sec vs batch size ------------------------------------
    ax3 = axes[2]
    for sl in sorted(valid["seq_len"].unique()):
        sub = valid[valid["seq_len"] == sl].sort_values("batch_size")
        ax3.plot(sub["batch_size"], sub["tokens_per_sec"], marker="s", label=f"S={sl}")
    ax3.set_title("Throughput vs Batch Size")
    ax3.set_xlabel("Batch size")
    ax3.set_ylabel("Tokens / second")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, "results/plots/latency_benchmark.png")
    plt.show()
    print("Plots saved to results/plots/latency_benchmark.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prompt(target_token_len: int, tokenizer) -> str:
    """Generate a dummy prompt that tokenises to approximately target_token_len tokens."""
    words = " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        for _ in range(target_token_len * 2)
    )
    ids = tokenizer.encode(words)[:target_token_len]
    return tokenizer.decode(ids)


def _nan_record(seq_len: int, batch_size: int) -> dict:
    return {
        "seq_len":        seq_len,
        "batch_size":     batch_size,
        "prefill_ms":     float("nan"),
        "e2e_ms":         float("nan"),
        "per_tok_ms":     float("nan"),
        "p90_per_tok_ms": float("nan"),
        "tokens_per_sec": float("nan"),
        "oom":            True,
    }


def _is_oom(e: Exception) -> bool:
    """Return True if the exception looks like a GPU OOM."""
    msg = str(e).lower()
    return isinstance(e, torch.cuda.OutOfMemoryError) or any(
        kw in msg for kw in ("out of memory", "cublas_status_execution_failed",
                             "cuda error", "cudaerrorillegaladdress")
    )


def _recover_gpu() -> None:
    """Free cached GPU memory after an OOM."""
    gc.collect()
    torch.cuda.empty_cache()


def _save_csv(df: pd.DataFrame, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Results saved → {path}")


def _save_fig(fig, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
