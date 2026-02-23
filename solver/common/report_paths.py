# -*- coding: utf-8 -*-
"""
Path utilities for job-scoped reporting directories.

Resolves a consistent layout for all algorithms:
    <prefix>/reports/<algorithm>/<material>/<JOBID or pidX>/
    └── tmp/    # intermediates like y_mem_result.csv

Priority for <algorithm>, <material>:
  1) $ALGORITHM / $MATERIAL (if set),
  2) inferred from REPORT_PATH if it contains ".../reports/<algorithm>/<material>/...",
  3) defaults: algorithm="PowerIteration", material="UnknownMaterial".

The <prefix> is preserved from REPORT_PATH (absolute or relative) up to "reports".
"""

from __future__ import annotations
import os
from pathlib import Path

__all__ = ["main_report_dir", "tmp_dir", "report_file_path"]

def _resolve_algo_material_from_report_path(base: Path):
    """Best-effort inference of <algorithm>/<material> from REPORT_PATH."""
    parts = base.parts
    if "reports" in parts:
        i = parts.index("reports")
        algo = parts[i + 1] if i + 1 < len(parts) else None
        mat  = parts[i + 2] if i + 2 < len(parts) else None
        return (algo, mat)
    # Fallback: if ALGORITHM appears somewhere, take the next part as material.
    algo_env = (os.environ.get("ALGORITHM") or "").strip()
    if algo_env and algo_env in parts:
        j = parts.index(algo_env)
        mat = parts[j + 1] if j + 1 < len(parts) else None
        return (algo_env, mat)
    return (None, None)

def _reports_prefix_from_report_path(base: Path) -> Path:
    """Preserve the path prefix up to and including 'reports', else return 'reports'."""
    parts = base.parts
    if "reports" in parts:
        k = parts.index("reports")
        return Path(*parts[: k + 1]) if k >= 0 else Path("reports")
    return Path("reports")

def main_report_dir() -> Path:
    """
    Return <prefix>/reports/<algorithm>/<material>/<JOBID or pidX>/ and create it.

    REPORT_PATH may be a file; if so, its parent is used to infer the folder shape.
    """
    report_path = os.environ.get("REPORT_PATH", "reports")
    base = Path(report_path)
    if base.suffix:
        base = base.parent

    # Preserve absolute/relative prefix
    root_prefix = _reports_prefix_from_report_path(base)

    # Resolve algorithm/material
    algo_env = (os.environ.get("ALGORITHM") or "").strip() or None
    mat_env  = (os.environ.get("MATERIAL")  or "").strip() or None
    algo_inf, mat_inf = _resolve_algo_material_from_report_path(base)

    algorithm = (algo_env or algo_inf or "PowerIteration")
    material  = (mat_env  or mat_inf  or "UnknownMaterial")

    # Job id
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    sub = f"{job_id}" if job_id else f"pid{os.getpid()}"

    out_dir = root_prefix / algorithm / material / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def tmp_dir() -> Path:
    """Return/create the per-job tmp/ folder under the main report directory."""
    d = main_report_dir() / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d

def report_file_path() -> Path:
    """
    Path to the human-readable run report file:
        main_report_dir() / <report_name>.txt

    <report_name> comes from:
      1) $REPORT_NAME, else
      2) basename(REPORT_PATH) if set, else
      3) 'report.txt'.
    """
    base = main_report_dir()
    report_name = os.environ.get("REPORT_NAME")
    if not report_name:
        rp = os.environ.get("REPORT_PATH")
        report_name = Path(rp).name if rp else "report.txt"
        if not report_name:
            report_name = "report.txt"
    if not report_name.lower().endswith(".txt"):
        report_name += ".txt"
    return base / report_name
