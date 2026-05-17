"""
benchmarks/throughput_test.py
-----------------------------
Sustained throughput benchmark: tokens/sec over a fixed duration window,
across batch sizes and sequence lengths.
"""

from __future__ import annotations

import time
from typing import List, Dict

import torch
import pandas as pd
import matplotlib.pyplot as plt

from benchmarks.latency_test import _make_prompt, _save_csv, _save_fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_throughput_benchmark(
    decoder,
    batch_sizes:  List[int] = (1, 2, 4, 8, 16),
    seq_len:      int       = 512,
    duration_sec: float     = 30.0,
    max_new_tokens: int     = 64,
) -> pd.DataFrame:
    """
    For each batch size, run as many generate_batch() calls as possible
    within duration_sec seconds and compute sustained tokens/sec.

    Parameters
    ----------
    decoder        : runtime.decoder.Decoder
    batch_sizes    : batch sizes to sweep
    seq_len        : fixed input prompt length for all trials
    duration_sec   : wall-clock window per configuration
    max_new_tokens : tokens generated per request per batch call

    Returns
    -------
    DataFrame with columns: batch_size, tokens_per_sec, requests_per_sec,
                            gpu_util_pct (placeholder), latency_ms_mean
    """
    records = []

    for batch_size in batch_sizes:
        print(f"  batch_size={batch_size}  seq_len={seq_len}  [{duration_sec}s window]")
        prompts = [_make_prompt(seq_len, decoder.tokenizer) for _ in range(batch_size)]

        # Warm-up
        decoder.generate_batch(prompts, max_new_tokens=8)
        torch.cuda.synchronize()

        total_tokens   = 0
        total_requests = 0
        latencies_ms   = []
        deadline       = time.perf_counter() + duration_sec

        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            decoder.generate_batch(prompts, max_new_tokens=max_new_tokens)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            total_tokens   += batch_size * max_new_tokens
            total_requests += batch_size
            latencies_ms.append(elapsed_ms)

        elapsed_total = sum(latencies_ms) / 1000.0   # sec
        tps           = total_tokens   / elapsed_total
        rps           = total_requests / elapsed_total
        mean_lat      = sum(latencies_ms) / len(latencies_ms)
        p99_lat       = sorted(latencies_ms)[int(0.99 * len(latencies_ms))]

        print(f"    → {tps:.1f} tok/s  |  {rps:.1f} req/s  |  lat_mean={mean_lat:.1f}ms")

        records.append({
            "batch_size":       batch_size,
            "seq_len":          seq_len,
            "tokens_per_sec":   tps,
            "requests_per_sec": rps,
            "latency_ms_mean":  mean_lat,
            "latency_ms_p99":   p99_lat,
            "total_tokens":     total_tokens,
            "num_batches":      len(latencies_ms),
        })

    df = pd.DataFrame(records)
    _save_csv(df, "results/csv_logs/throughput_results.csv")
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_throughput_results(df: pd.DataFrame) -> None:
    """
    Two-panel plot:
      Left  : tokens/sec vs batch size
      Right : mean latency vs batch size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Sustained Throughput — seq_len={df['seq_len'].iloc[0]}",
        fontsize=13, fontweight="bold"
    )

    # ---- Tokens/sec
    ax1.bar(df["batch_size"].astype(str), df["tokens_per_sec"], color="#2196F3", edgecolor="white")
    ax1.set_title("Tokens / Second")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Tokens/sec")
    for i, row in df.iterrows():
        ax1.text(i, row["tokens_per_sec"] * 1.02, f"{row['tokens_per_sec']:.0f}",
                 ha="center", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ---- Latency
    ax2.errorbar(
        df["batch_size"], df["latency_ms_mean"],
        yerr=df["latency_ms_p99"] - df["latency_ms_mean"],
        fmt="o-", color="#F44336", capsize=5, label="mean ± (p99-mean)"
    )
    ax2.set_title("Batch Latency (ms)")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Latency (ms)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, "results/plots/throughput_benchmark.png")
    plt.show()
    print("Plots saved to results/plots/throughput_benchmark.png")
