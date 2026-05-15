#!/usr/bin/env python3
"""
validate_env.py
Environment validation script for the LLM Inference Engine on Jetson Orin Nano (sm_87).
Checks Python, CUDA, Triton, and all required libraries.
Run inside the container: python3 validate_env.py
"""

import sys
import os
import time
import importlib

# ── ANSI colors ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"

results = []

def check(label, fn):
    try:
        msg = fn()
        print(f"  {PASS} {label}: {msg}")
        results.append((label, True, msg))
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        results.append((label, False, str(e)))

def section(title):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")


# ── 1. Python ─────────────────────────────────────────────────
section("1. Python Runtime")

check("Python version ≥ 3.10",
      lambda: (v := sys.version_info, __import__('sys').exit(1)
               if v < (3, 10) else f"{v.major}.{v.minor}.{v.micro}")[1])

# simpler lambda that works:
def check_python():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"{v.major}.{v.minor} is below 3.10")
    return f"{v.major}.{v.minor}.{v.micro}"

results.clear()  # reset before real run
check("Python version ≥ 3.10", check_python)

check("sys.platform (expect linux)",
      lambda: sys.platform if sys.platform == "linux" else (_ for _ in ()).throw(RuntimeError(sys.platform)))

check("Architecture (expect aarch64)",
      lambda: (a := os.uname().machine, a if "aarch64" in a else (_ for _ in ()).throw(RuntimeError(a)))[1])


# ── 2. CUDA Environment ───────────────────────────────────────
section("2. CUDA Environment")

check("CUDA_HOME set",
      lambda: os.environ["CUDA_HOME"])

check("nvcc on PATH", lambda: (
    __import__('subprocess').check_output(["nvcc","--version"]).decode()
    .split("release")[-1].strip().split(",")[0].strip()
))

check("TORCH_CUDA_ARCH_LIST=8.7",
      lambda: (v := os.environ.get("TORCH_CUDA_ARCH_LIST",""),
               v if v == "8.7" else (_ for _ in ()).throw(RuntimeError(f"got '{v}'")))[1])

check("LD_LIBRARY_PATH includes cuda/lib64",
      lambda: (p := os.environ.get("LD_LIBRARY_PATH",""),
               p if "cuda/lib64" in p else (_ for _ in ()).throw(RuntimeError("not found")))[1])


# ── 3. PyTorch ────────────────────────────────────────────────
section("3. PyTorch")

import torch  # noqa – fail loudly if missing

check("torch importable",
      lambda: torch.__version__)

check("CUDA available",
      lambda: (torch.cuda.is_available() or (_ for _ in ()).throw(RuntimeError("cuda not available")),
               torch.version.cuda)[1])

check("Device is sm_87",
      lambda: (cc := torch.cuda.get_device_capability(0),
               f"sm_{cc[0]}{cc[1]}" if cc == (8, 7) else
               (_ for _ in ()).throw(RuntimeError(f"sm_{cc[0]}{cc[1]} (expected sm_87")))[1])

check("Device name",
      lambda: torch.cuda.get_device_name(0))

check("VRAM reported (MB)",
      lambda: f"{torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

check("FP16 tensor creation on GPU", lambda: (
    t := torch.ones(4, 4, dtype=torch.float16, device="cuda"),
    f"shape={tuple(t.shape)} dtype={t.dtype}"
)[1])

check("BF16 tensor creation on GPU", lambda: (
    t := torch.ones(4, 4, dtype=torch.bfloat16, device="cuda"),
    f"shape={tuple(t.shape)} dtype={t.dtype}"
)[1])

check("Small GEMM on GPU (FP16)", lambda: (
    a := torch.randn(64, 64, dtype=torch.float16, device="cuda"),
    b := torch.randn(64, 64, dtype=torch.float16, device="cuda"),
    c := torch.matmul(a, b),
    f"output shape={tuple(c.shape)}"
)[3])

check("cuBLAS workspace config env",
      lambda: os.environ.get("CUBLAS_WORKSPACE_CONFIG", "NOT SET"))

check("Tensor Core GEMM (torch.backends.cuda.matmul.allow_tf32)",
      lambda: (torch.backends.cuda.matmul.allow_tf32, "tf32 allowed" if torch.backends.cuda.matmul.allow_tf32 else "tf32 off (expected for sm_87 Ampere)")[1])


# ── 4. Triton ─────────────────────────────────────────────────
section("4. Triton")

def check_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
        pid = tl.program_id(0)
        x = tl.load(x_ptr + pid)
        y = tl.load(y_ptr + pid)
        tl.store(out_ptr + pid, x + y)

    N = 128
    x = torch.ones(N, device="cuda", dtype=torch.float32)
    y = torch.ones(N, device="cuda", dtype=torch.float32) * 2
    out = torch.empty(N, device="cuda", dtype=torch.float32)
    _add_kernel[(N,)](x, y, out, N)
    torch.cuda.synchronize()
    assert float(out[0]) == 3.0, f"expected 3.0, got {float(out[0])}"
    return f"v{triton.__version__} — kernel executed OK"

check("Triton import + JIT kernel", check_triton)

check("TRITON_CACHE_DIR set",
      lambda: os.environ.get("TRITON_CACHE_DIR", "NOT SET"))


# ── 5. Core Python packages ───────────────────────────────────
section("5. Core Python Packages")

REQUIRED_PACKAGES = [
    "transformers",
    "accelerate",
    "tokenizers",
    "numpy",
    "scipy",
    "einops",
    "sentencepiece",
    "tqdm",
    "psutil",
]

for pkg in REQUIRED_PACKAGES:
    def _chk(p=pkg):
        m = importlib.import_module(p)
        return getattr(m, "__version__", "imported (no __version__)")
    check(f"{pkg}", _chk)


# ── 6. Optional / Profiling packages ─────────────────────────
section("6. Optional / Profiling Packages")

OPTIONAL_PACKAGES = [
    ("matplotlib",  "plotting"),
    ("pandas",      "CSV logging"),
    ("pynvml",      "NVML GPU metrics"),
    ("nvtx",        "Nsight NVTX markers"),
]

for pkg, purpose in OPTIONAL_PACKAGES:
    def _opt(p=pkg, desc=purpose):
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "ok")
        return f"v{v} ({desc})"
    try:
        msg = _opt()
        print(f"  {PASS} {pkg}: {msg}")
        results.append((pkg, True, msg))
    except ImportError:
        print(f"  {WARN} {pkg}: not installed — {purpose} unavailable")
        results.append((pkg, None, "optional, missing"))


# ── 7. CUTLASS headers ────────────────────────────────────────
section("7. CUTLASS Headers")

check("CUTLASS_PATH set",
      lambda: os.environ.get("CUTLASS_PATH", "NOT SET"))

check("CUTLASS include dir exists", lambda: (
    p := os.path.join(os.environ.get("CUTLASS_PATH", ""), "include"),
    p if os.path.isdir(p) else (_ for _ in ()).throw(RuntimeError(f"missing: {p}"))
)[1])

check("gemm.h present in CUTLASS", lambda: (
    f := os.path.join(os.environ.get("CUTLASS_PATH", ""), "include", "cutlass", "gemm", "gemm.h"),
    f if os.path.isfile(f) else (_ for _ in ()).throw(RuntimeError(f"not found: {f}"))
)[1])


# ── 8. cuSPARSELt ────────────────────────────────────────────
section("8. cuSPARSELt")

check("cuSPARSELt lib dir exists",
      lambda: (p := "/opt/cusparselt/lib",
               p if os.path.isdir(p) else (_ for _ in ()).throw(RuntimeError("missing")))[1])

check("libcusparseLt.so present", lambda: (
    import_os := os,
    libs := [f for f in import_os.listdir("/opt/cusparselt/lib") if "cusparseLt" in f],
    libs[0] if libs else (_ for _ in ()).throw(RuntimeError("no libcusparseLt found"))
)[2])


# ── 9. Workspace layout ───────────────────────────────────────
section("9. Workspace Directory Layout")

EXPECTED_DIRS = [
    "/workspace/kernels",
    "/workspace/runtime",
    "/workspace/memory",
    "/workspace/benchmarks",
    "/workspace/profiling",
    "/workspace/results/plots",
    "/workspace/results/csv_logs",
    "/workspace/profiling/nsight_reports",
    "/workspace/.triton_cache",
]

for d in EXPECTED_DIRS:
    def _dir(path=d):
        if os.path.isdir(path):
            return "exists"
        raise RuntimeError(f"missing — run container from project root")
    check(d, _dir)


# ── 10. Quick bandwidth micro-benchmark ───────────────────────
section("10. GPU Memory Bandwidth Smoke-Test")

def bw_test():
    SIZE = 256 * 1024 * 1024 // 4  # 256 MB in float32 elements
    src = torch.empty(SIZE, dtype=torch.float32, device="cuda")
    dst = torch.empty(SIZE, dtype=torch.float32, device="cuda")

    # warm-up
    dst.copy_(src)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / 5
    bytes_transferred = SIZE * 4 * 2  # read + write
    bw_gbs = bytes_transferred / elapsed / 1e9
    return f"{bw_gbs:.1f} GB/s (Orin Nano peak ≈ 40–50 GB/s)"

check("D2D copy bandwidth", bw_test)


# ── Summary ───────────────────────────────────────────────────
section("Summary")

passed  = sum(1 for _, s, _ in results if s is True)
failed  = sum(1 for _, s, _ in results if s is False)
warned  = sum(1 for _, s, _ in results if s is None)
total   = len(results)

print(f"\n  Total checks : {total}")
print(f"  {GREEN}Passed{RESET}       : {passed}")
print(f"  {YELLOW}Warnings{RESET}     : {warned}  (optional packages)")
print(f"  {RED}Failed{RESET}       : {failed}")

if failed == 0:
    print(f"\n  {GREEN}{BOLD}✓ Environment looks good. Ready to build and run inference.{RESET}")
else:
    print(f"\n  {RED}{BOLD}✗ {failed} check(s) failed. Fix the issues above before proceeding.{RESET}")
    print(f"\n  Failed checks:")
    for label, status, msg in results:
        if status is False:
            print(f"    • {label}: {msg}")

print()