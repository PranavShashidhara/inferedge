"""
profiling/roofline_analysis.py
------------------------------
Roofline model visualisation for GPU kernel analysis.

Given hardware ceilings and per-kernel (FLOPs, bytes) measurements,
plots each kernel on the roofline and classifies it as compute-bound
or memory-bound.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Roofline Analyser
# ---------------------------------------------------------------------------

class RooflineAnalyzer:
    
    def __init__(
        self,
        peak_flops_tflops: float = 1.4,
        peak_bw_gbps:      float = 51.2,
    ):
        self.peak_flops  = peak_flops_tflops * 1e12   # FLOP/s
        self.peak_bw     = peak_bw_gbps      * 1e9    # byte/s

        # Ridge point: arithmetic intensity where compute == memory limits
        self.ridge_point = self.peak_flops / self.peak_bw   # FLOP/byte

    # ------------------------------------------------------------------
    # Main plot
    # ------------------------------------------------------------------

    def plot_roofline(
        self,
        kernel_stats: List[Dict[str, Any]],
        title: str = "Roofline Model — Jetson Orin Nano (sm_87)",
        save_path: Optional[str] = "results/plots/roofline.png",
    ) -> None:
        """
        Render the roofline diagram.

        Parameters
        ----------
        kernel_stats : list of dicts with keys:
                       phase, flops_estimate, bytes_estimate, latency_ms
                       (produced by runtime.decoder.Decoder._decode_loop)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title, fontsize=13, fontweight="bold")

        # ---- Roofline ceiling ----------------------------------------
        ai_range = np.logspace(-2, 4, 500)   # FLOP/byte
        perf_roof = np.minimum(self.peak_bw * ai_range, self.peak_flops)
        ax.plot(ai_range, perf_roof / 1e9,   # → GFLOP/s
                color="#212121", linewidth=2.5, label="Roofline ceiling")

        # Dashed ridge line
        ax.axvline(self.ridge_point, color="grey", linestyle="--", alpha=0.6,
                   label=f"Ridge point ({self.ridge_point:.1f} FLOP/byte)")

        # Bandwidth and compute labels
        ax.text(self.ridge_point * 0.05, self.peak_flops / 1e9 * 0.6,
                f"BW-bound\n≤{self.peak_bw/1e9:.0f} GB/s", color="#1565C0", fontsize=9)
        ax.text(self.ridge_point * 3, self.peak_flops / 1e9 * 0.95,
                f"Compute-bound\n≤{self.peak_flops/1e12:.1f} TFLOPS", color="#B71C1C", fontsize=9)

        # ---- Kernel points -------------------------------------------
        colors = {"prefill": "#1E88E5", "decode": "#E53935"}
        markers = {"prefill": "^", "decode": "o"}

        plotted_labels: set = set()
        for stat in kernel_stats:
            if stat.get("bytes_estimate", 0) == 0:
                continue

            ai   = stat["flops_estimate"] / stat["bytes_estimate"]   # FLOP/byte
            perf = stat["flops_estimate"] / (stat["latency_ms"] * 1e-3) / 1e9  # GFLOP/s

            phase  = stat.get("phase", "decode")
            color  = colors.get(phase, "#43A047")
            marker = markers.get(phase, "o")
            label  = phase if phase not in plotted_labels else None
            plotted_labels.add(phase)

            ax.scatter(ai, perf, color=color, marker=marker, s=60,
                       zorder=5, label=label, alpha=0.85)

        # ---- Formatting ----------------------------------------------
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
        ax.set_ylabel("Performance (GFLOP/s)", fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=9)

        # Annotate attainment
        self._annotate_attainment(ax, kernel_stats)

        plt.tight_layout()
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Roofline plot saved to {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary(self, kernel_stats: List[Dict[str, Any]]) -> None:
        """Print a text table of per-kernel roofline analysis."""
        print(f"\n{'='*65}")
        print(f"  Roofline Summary  (ridge={self.ridge_point:.1f} FLOP/byte)")
        print(f"{'='*65}")
        print(f"  {'Phase':<10} {'Step':<6} {'AI':>10} {'Perf(G)':>10} {'Bound':<15}")
        print(f"  {'-'*55}")

        for stat in kernel_stats:
            if stat.get("bytes_estimate", 0) == 0:
                continue
            ai    = stat["flops_estimate"] / stat["bytes_estimate"]
            perf  = stat["flops_estimate"] / (stat["latency_ms"] * 1e-3) / 1e9
            bound = "Memory-bound" if ai < self.ridge_point else "Compute-bound"
            step  = stat.get("step", "-")
            phase = stat.get("phase", "?")
            print(f"  {phase:<10} {str(step):<6} {ai:>10.2f} {perf:>10.2f} {bound:<15}")

        print(f"{'='*65}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _annotate_attainment(
        self, ax, kernel_stats: List[Dict[str, Any]]
    ) -> None:
        """Add text showing average % of roofline attained."""
        attainments = []
        for stat in kernel_stats:
            if stat.get("bytes_estimate", 0) == 0:
                continue
            ai   = stat["flops_estimate"] / stat["bytes_estimate"]
            perf = stat["flops_estimate"] / (stat["latency_ms"] * 1e-3)
            ceil = min(self.peak_bw * ai, self.peak_flops)
            attainments.append(perf / ceil * 100)

        if attainments:
            mean_att = sum(attainments) / len(attainments)
            ax.text(
                0.02, 0.04,
                f"Mean roofline attainment: {mean_att:.1f}%",
                transform=ax.transAxes,
                fontsize=9, color="#555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

    def classify(self, flops: float, bytes_accessed: float) -> str:
        """Return 'memory-bound' or 'compute-bound' for a kernel."""
        ai = flops / bytes_accessed if bytes_accessed > 0 else float("inf")
        return "compute-bound" if ai >= self.ridge_point else "memory-bound"
