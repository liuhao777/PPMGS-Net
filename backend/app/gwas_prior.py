from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASSEL_DOWNLOAD_URL = "https://tassel.bitbucket.io/installer/TASSEL_5_windows-x64.exe"
TASSEL_DOC_URL = "https://tassel.bitbucket.io/"
PRIOR_SPARSITY_ALIASES = {
    "none": "none",
    "": "none",
    "top_0_5pct": "top_0_5pct",
    "top_0.5pct": "top_0_5pct",
    "top0.5": "top_0_5pct",
    "top_1pct": "top_1pct",
    "top_1%": "top_1pct",
    "top1": "top_1pct",
    "top_5pct": "top_5pct",
    "top_5%": "top_5pct",
    "top5": "top_5pct",
    "p_1e-3": "p_1e-3",
    "p<1e-3": "p_1e-3",
    "p_lt_1e-3": "p_1e-3",
    "p_1e-4": "p_1e-4",
    "p<1e-4": "p_1e-4",
    "p_lt_1e-4": "p_1e-4",
}


@dataclass
class TasselPriorResult:
    prior_scores: np.ndarray
    summary: dict[str, object]
    prior_table: pd.DataFrame


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _normalize_prior_sparsity(mode: str | None) -> str:
    key = str(mode or "none").strip().lower().replace(" ", "_")
    return PRIOR_SPARSITY_ALIASES.get(key, "none")


def _prior_sparsity_spec(mode: str) -> dict[str, object]:
    mode = _normalize_prior_sparsity(mode)
    if mode == "top_0_5pct":
        return {"mode": mode, "kind": "top_fraction", "fraction": 0.005}
    if mode == "top_1pct":
        return {"mode": mode, "kind": "top_fraction", "fraction": 0.01}
    if mode == "top_5pct":
        return {"mode": mode, "kind": "top_fraction", "fraction": 0.05}
    if mode == "p_1e-3":
        return {"mode": mode, "kind": "min_score", "min_score": 3.0, "pvalue_threshold": 1e-3}
    if mode == "p_1e-4":
        return {"mode": mode, "kind": "min_score", "min_score": 4.0, "pvalue_threshold": 1e-4}
    return {"mode": "none", "kind": "none"}


def _apply_prior_sparsity(scores: np.ndarray, mode: str | None) -> tuple[np.ndarray, dict[str, object]]:
    mode = _normalize_prior_sparsity(mode)
    values = np.asarray(scores, dtype=np.float32)
    sparse = np.zeros_like(values, dtype=np.float32)
    before = [int(np.sum(row > 0)) for row in values]
    spec = _prior_sparsity_spec(mode)
    if mode == "none":
        sparse = values.copy()
    elif spec["kind"] == "top_fraction":
        fraction = float(spec["fraction"])
        keep_count = max(1, int(np.ceil(values.shape[1] * fraction))) if values.shape[1] else 0
        for trait_idx, row in enumerate(values):
            positive_idx = np.flatnonzero(row > 0)
            if positive_idx.size == 0:
                continue
            selected = positive_idx[np.argsort(row[positive_idx])[::-1][: min(keep_count, positive_idx.size)]]
            sparse[trait_idx, selected] = row[selected]
    elif spec["kind"] == "min_score":
        sparse = np.where(values >= float(spec["min_score"]), values, 0.0).astype(np.float32)
    after = [int(np.sum(row > 0)) for row in sparse]
    return sparse.astype(np.float32), {
        "mode": mode,
        "spec": spec,
        "applied": mode != "none",
        "nonzero_before_by_trait": before,
        "nonzero_after_by_trait": after,
        "nonzero_removed_by_trait": [int(max(0, b - a)) for b, a in zip(before, after)],
    }


def _safe_name(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text or "value"


def _unique_safe_names(values: Iterable[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for value in values:
        base = _safe_name(value)
        count = counts.get(base, 0)
        counts[base] = count + 1
        result.append(base if count == 0 else f"{base}_{count + 1}")
    return result


def _quote_for_bat(value: str | Path) -> str:
    text = str(value)
    return f'"{text}"' if any(char.isspace() for char in text) else text


def resolve_tassel_pipeline(tassel_pipeline_path: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if tassel_pipeline_path:
        candidates.append(Path(tassel_pipeline_path))
    env_value = os.environ.get("TASSEL5_RUN_PIPELINE") or os.environ.get("TASSEL_RUN_PIPELINE")
    if env_value:
        candidates.append(Path(env_value))

    for root in (
        Path("C:/Program Files/TASSEL 5"),
        Path("C:/Program Files (x86)/TASSEL 5"),
        PROJECT_ROOT / "external" / "tassel5",
        PROJECT_ROOT / "tools" / "tassel5",
    ):
        candidates.extend(
            [
                root / "run_pipeline.bat",
                root / "run_pipeline.pl",
                root / "tassel-5-standalone" / "run_pipeline.bat",
                root / "tassel-5-standalone" / "run_pipeline.pl",
            ]
        )

    for name in ("run_pipeline.bat", "run_pipeline.pl", "run_pipeline"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _infer_marker_locus(marker: str, index: int) -> tuple[str, int]:
    text = str(marker)
    patterns = [
        r"(?:^|[^0-9A-Za-z])chr(?:omosome)?[_:\- ]*([0-9A-Za-z]+)[_:\- ]+([0-9]+)",
        r"^([0-9A-Za-z]+)[:_|\-]([0-9]{3,})$",
        r"^S([0-9A-Za-z]+)[_:|\-]([0-9]{3,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return str(match.group(1)), int(match.group(2))
            except ValueError:
                continue
    return "1", int(index) + 1


def _impute_numeric_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32).copy()
    fill_values = np.zeros(matrix.shape[1], dtype=np.float32)
    for col_idx in range(matrix.shape[1]):
        column = matrix[:, col_idx]
        observed = column[np.isfinite(column)]
        if observed.size == 0:
            fill = 0.0
        else:
            rounded = np.rint(observed)
            uniq, counts = np.unique(rounded, return_counts=True)
            fill = float(uniq[int(np.argmax(counts))])
        fill_values[col_idx] = fill
        matrix[~np.isfinite(matrix[:, col_idx]), col_idx] = fill
    return matrix, fill_values


def _hapmap_call(value: float, binary_mode: bool) -> str:
    if not np.isfinite(value):
        return "NN"
    rounded = float(np.rint(value))
    if binary_mode:
        return "AA" if rounded <= 0.0 else "TT"
    if rounded <= 0.0:
        return "AA"
    if rounded >= 2.0:
        return "TT"
    return "AT"


def write_tassel_hapmap(
    x_raw: np.ndarray,
    sample_ids: list[str],
    marker_names: list[str],
    output_path: Path,
    force_synthetic_positions: bool = False,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_imputed, fill_values = _impute_numeric_matrix(x_raw)
    finite = x_imputed[np.isfinite(x_imputed)]
    binary_mode = bool(finite.size and np.nanmax(finite) <= 1.1)

    header = [
        "rs#",
        "alleles",
        "chrom",
        "pos",
        "strand",
        "assembly#",
        "center",
        "protLSID",
        "assayLSID",
        "panel",
        "QCcode",
        *sample_ids,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for marker_idx, marker in enumerate(marker_names):
            if force_synthetic_positions:
                chrom, pos = "1", marker_idx + 1
            else:
                chrom, pos = _infer_marker_locus(marker, marker_idx)
            row = [
                str(marker),
                "A/T",
                str(chrom),
                str(pos),
                "+",
                "NA",
                "PPMGS-Net",
                "NA",
                "NA",
                "PPMGS-Net",
                "NA",
            ]
            row.extend(_hapmap_call(value, binary_mode=binary_mode) for value in x_imputed[:, marker_idx])
            handle.write("\t".join(row) + "\n")
    return {
        "path": str(output_path),
        "format": "TASSEL HapMap",
        "binary_mode": binary_mode,
        "marker_count": int(len(marker_names)),
        "sample_count": int(len(sample_ids)),
        "imputation": "marker_mode_before_hapmap_export",
        "position_mode": (
            "synthetic_input_order"
            if force_synthetic_positions
            else "inferred_from_marker_name_with_fallback"
        ),
        "fill_value_min": float(np.min(fill_values)) if fill_values.size else None,
        "fill_value_max": float(np.max(fill_values)) if fill_values.size else None,
    }


def _compute_marker_pcs(x_raw: np.ndarray, pc_count: int) -> np.ndarray:
    pc_count = max(0, int(pc_count))
    if pc_count == 0:
        return np.zeros((x_raw.shape[0], 0), dtype=np.float32)
    x_imputed, _ = _impute_numeric_matrix(x_raw)
    x_centered = x_imputed - x_imputed.mean(axis=0, keepdims=True)
    x_std = x_centered.std(axis=0, keepdims=True)
    x_std = np.where(x_std > 1e-6, x_std, 1.0)
    z = x_centered / x_std
    kinship = (z.astype(np.float64) @ z.astype(np.float64).T) / max(z.shape[1], 1)
    kinship = (kinship + kinship.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(kinship + 1e-8 * np.eye(kinship.shape[0], dtype=np.float64))
    order = np.argsort(eigvals)[::-1][: min(pc_count, eigvecs.shape[1])]
    pcs = eigvecs[:, order] * np.sqrt(np.maximum(eigvals[order], 0.0))[None, :]
    if pcs.shape[1] < pc_count:
        pcs = np.pad(pcs, ((0, 0), (0, pc_count - pcs.shape[1])))
    return pcs.astype(np.float32)


def write_tassel_phenotype(
    y_raw: np.ndarray,
    y_mask: np.ndarray,
    sample_ids: list[str],
    trait_names: list[str],
    x_raw: np.ndarray,
    output_path: Path,
    pc_count: int = 3,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_raw, dtype=np.float32)
    mask = np.asarray(y_mask, dtype=np.float32) > 0
    pcs = _compute_marker_pcs(x_raw, pc_count=pc_count)
    pc_names = [f"PC{idx + 1}" for idx in range(pcs.shape[1])]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("<Phenotype>\n")
        type_row = ["taxa", *(["data"] * len(trait_names)), *(["covariate"] * len(pc_names))]
        name_row = ["Taxa", *trait_names, *pc_names]
        handle.write("\t".join(type_row) + "\n")
        handle.write("\t".join(name_row) + "\n")
        for row_idx, sample in enumerate(sample_ids):
            values = [str(sample)]
            for trait_idx in range(len(trait_names)):
                if mask[row_idx, trait_idx] and np.isfinite(y[row_idx, trait_idx]):
                    values.append(f"{float(y[row_idx, trait_idx]):.10g}")
                else:
                    values.append("NaN")
            values.extend(f"{float(value):.10g}" for value in pcs[row_idx])
            handle.write("\t".join(values) + "\n")
    return {
        "path": str(output_path),
        "format": "TASSEL Phenotype",
        "pc_covariates": pc_names,
        "trait_count": int(len(trait_names)),
        "sample_count": int(len(sample_ids)),
        "observed_by_trait": {
            trait: int(np.sum(mask[:, idx]))
            for idx, trait in enumerate(trait_names)
        },
    }


def write_tassel_kinship(x_raw: np.ndarray, sample_ids: list[str], output_path: Path) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_imputed, _ = _impute_numeric_matrix(x_raw)
    x_centered = x_imputed - x_imputed.mean(axis=0, keepdims=True)
    x_std = x_centered.std(axis=0, keepdims=True)
    x_std = np.where(x_std > 1e-6, x_std, 1.0)
    z = x_centered / x_std
    kinship = (z.astype(np.float64) @ z.astype(np.float64).T) / max(z.shape[1], 1)
    kinship = (kinship + kinship.T) * 0.5
    mean_diag = float(np.mean(np.diag(kinship))) if kinship.size else 1.0
    if np.isfinite(mean_diag) and mean_diag > 1e-8:
        kinship = kinship / mean_diag

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(str(len(sample_ids)) + "\t" * len(sample_ids) + "\n")
        for row_idx, sample in enumerate(sample_ids):
            row = [str(sample)]
            row.extend(f"{float(value):.10g}" for value in kinship[row_idx])
            handle.write("\t".join(row) + "\n")
    return {
        "path": str(output_path),
        "format": "TASSEL square kinship matrix",
        "method": "centered_scaled_marker_grm",
        "mean_diagonal": float(np.mean(np.diag(kinship))) if kinship.size else None,
        "sample_count": int(len(sample_ids)),
    }


def _build_tassel_command(
    pipeline_path: Path,
    hapmap_path: Path,
    phenotype_path: Path,
    kinship_path: Path,
    mlm_output_prefix: Path,
    max_p: float | None = None,
) -> list[str]:
    java_mem_min = os.environ.get("TASSEL_JAVA_MEM_MIN", "-Xms4g").strip() or "-Xms4g"
    java_mem_max = os.environ.get("TASSEL_JAVA_MEM_MAX", "-Xmx64g").strip() or "-Xmx64g"
    command = [
        str(pipeline_path),
        java_mem_min,
        java_mem_max,
        "-fork1",
        "-h",
        str(hapmap_path),
        "-fork2",
        "-t",
        str(phenotype_path),
        "-fork3",
        "-k",
        str(kinship_path),
        "-combine4",
        "-input1",
        "-input2",
        "-intersect",
        "-combine5",
        "-input4",
        "-input3",
        "-mlm",
        "-mlmVarCompEst",
        "P3D",
        "-mlmCompressionLevel",
        "None",
        "-mlmOutputFile",
        str(mlm_output_prefix),
    ]
    if max_p is not None and max_p > 0:
        command.extend(["-mlmMaxP", str(float(max_p))])
    return command


def _write_command_script(command: list[str], output_path: Path, working_dir: Path | None = None) -> None:
    command_line = " ".join(_quote_for_bat(part) for part in command)
    if working_dir is not None:
        output_path.write_text(
            "@echo off\n"
            "REM Run this file after TASSEL 5 and Java are installed.\n"
            f"pushd {_quote_for_bat(str(working_dir))}\n"
            f"{command_line}\n"
            "set TASSEL_EXITCODE=%ERRORLEVEL%\n"
            "popd\n"
            "exit /b %TASSEL_EXITCODE%\n",
            encoding="utf-8",
        )
        return
    output_path.write_text(
        "@echo off\n"
        "REM Run this file after TASSEL 5 and Java are installed.\n"
        + command_line
        + "\n",
        encoding="utf-8",
    )


def _detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    if "\t" in sample:
        return "\t"
    return ","


def _find_column(columns: Iterable[object], candidates: set[str], contains: tuple[str, ...] = ()) -> object | None:
    normalized = [(str(col).strip(), str(col).strip().lower().replace(" ", "_")) for col in columns]
    for original, lower in normalized:
        if lower in candidates:
            return original
    for original, lower in normalized:
        if contains and all(part in lower for part in contains):
            return original
    return None


def _find_pvalue_column(columns: Iterable[object]) -> object | None:
    normalized = [(str(col).strip(), str(col).strip().lower().replace(" ", "_")) for col in columns]
    exact = {
        "p",
        "pvalue",
        "p_value",
        "p-value",
        "p_marker",
        "marker_p",
        "pvalue_marker",
        "p_marker_f",
        "add_p",
        "additive_p",
        "marker_pvalue",
        "p_value_marker",
    }
    for original, lower in normalized:
        if lower in exact:
            return original
    for original, lower in normalized:
        compact = lower.replace("-", "_")
        if "pvalue" in compact or "p_value" in compact or compact.endswith("_p"):
            return original
    return None


def _find_position_column(columns: Iterable[object]) -> object | None:
    return _find_column(
        columns,
        {"pos", "position", "bp", "physical_position", "physical_pos"},
        contains=("pos",),
    )


def _parse_pvalue(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "null"}:
        return None
    text = text.replace(",", "")
    if text.startswith("<"):
        text = text[1:].strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def _normalize_trait_name(value: object, trait_lookup: dict[str, int]) -> tuple[str | None, int | None]:
    text = str(value).strip()
    lower = text.lower()
    if lower in trait_lookup:
        return text, trait_lookup[lower]
    for trait, idx in trait_lookup.items():
        if trait and trait in lower:
            return text, idx
    return None, None


def _infer_trait_from_path(path: Path, trait_names: list[str], trait_lookup: dict[str, int]) -> tuple[str | None, int | None]:
    lower_name = path.name.lower()
    for trait in trait_names:
        key = trait.lower()
        if key in lower_name:
            return trait, trait_lookup[key]
    return None, None


def _marker_index_from_row(
    row: pd.Series,
    marker_col: object | None,
    chrom_col: object | None,
    pos_col: object | None,
    marker_lookup: dict[str, int],
    locus_lookup: dict[tuple[str, int], int],
    marker_count: int,
) -> int | None:
    if marker_col is not None:
        marker = str(row[marker_col]).strip()
        if marker in marker_lookup:
            return marker_lookup[marker]
        marker_clean = marker.strip("'\"")
        if marker_clean in marker_lookup:
            return marker_lookup[marker_clean]
        try:
            numeric = float(marker_clean)
            if numeric.is_integer():
                site = int(numeric)
                if 1 <= site <= marker_count:
                    return site - 1
                if 0 <= site < marker_count:
                    return site
        except ValueError:
            pass

    if chrom_col is not None and pos_col is not None:
        chrom = str(row[chrom_col]).strip()
        pos_raw = pd.to_numeric(row[pos_col], errors="coerce")
        if pd.notna(pos_raw) and np.isfinite(float(pos_raw)):
            pos = int(round(float(pos_raw)))
            candidates = [
                (chrom, pos),
                (chrom.replace("chr", "", 1) if chrom.lower().startswith("chr") else chrom, pos),
                (chrom.lower().replace("chr", "", 1), pos),
            ]
            for candidate in candidates:
                if candidate in locus_lookup:
                    return locus_lookup[candidate]
    return None


def _candidate_tassel_outputs(output_dir: Path, prefix: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in output_dir.glob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "mlm" in name or prefix.name.lower() in name:
            candidates.append(path)
    if prefix.exists() and prefix.is_file():
        candidates.insert(0, prefix)
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_tassel_runtime_paths(
    hapmap_path: Path,
    phenotype_path: Path,
    kinship_path: Path,
    mlm_output_prefix: Path,
) -> tuple[Path, Path, Path, Path, Path | None]:
    tassel_arg_paths = [hapmap_path, phenotype_path, kinship_path, mlm_output_prefix]
    if not any(" " in str(path) for path in tassel_arg_paths):
        return hapmap_path, phenotype_path, kinship_path, mlm_output_prefix, None

    runtime_dir = Path(tempfile.mkdtemp(prefix="ppmgs_tassel_"))
    runtime_inputs = runtime_dir / "inputs"
    runtime_hapmap = runtime_inputs / hapmap_path.name
    runtime_phenotype = runtime_inputs / phenotype_path.name
    runtime_kinship = runtime_inputs / kinship_path.name
    runtime_mlm_output_prefix = runtime_dir / mlm_output_prefix.name
    _link_or_copy_file(hapmap_path, runtime_hapmap)
    _link_or_copy_file(phenotype_path, runtime_phenotype)
    _link_or_copy_file(kinship_path, runtime_kinship)
    return runtime_hapmap, runtime_phenotype, runtime_kinship, runtime_mlm_output_prefix, runtime_dir


def _copy_tassel_runtime_outputs(runtime_dir: Path | None, output_dir: Path) -> list[str]:
    if runtime_dir is None:
        return []
    copied: list[str] = []
    for path in runtime_dir.glob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "mlm" not in name and "output" not in name:
            continue
        dst = output_dir / path.name
        shutil.copy2(path, dst)
        copied.append(str(dst))
    (output_dir / "tassel_runtime_dir.txt").write_text(str(runtime_dir) + "\n", encoding="utf-8")
    return copied


def _parse_tassel_outputs(
    output_paths: list[Path],
    marker_names: list[str],
    trait_names: list[str],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    marker_lookup = {marker: idx for idx, marker in enumerate(marker_names)}
    locus_lookup = {_infer_marker_locus(marker, idx): idx for idx, marker in enumerate(marker_names)}
    trait_lookup = {trait.lower(): idx for idx, trait in enumerate(trait_names)}
    raw_scores = np.zeros((len(trait_names), len(marker_names)), dtype=np.float32)
    raw_pvalues = np.full((len(trait_names), len(marker_names)), np.nan, dtype=np.float32)
    parsed_rows: list[dict[str, object]] = []
    files_used: list[str] = []
    files_inspected: list[dict[str, object]] = []

    for path in output_paths:
        try:
            df = pd.read_csv(path, sep=_detect_delimiter(path), engine="python")
        except Exception:
            continue
        if df.empty:
            continue
        files_inspected.append(
            {
                "path": str(path),
                "rows": int(len(df)),
                "columns": [str(col) for col in df.columns],
            }
        )
        marker_col = _find_column(
            df.columns,
            {"marker", "markers", "site", "snp", "rs", "rs#"},
            contains=("marker",),
        )
        chrom_col = _find_column(df.columns, {"chr", "chrom", "chromosome"}, contains=("chr",))
        pos_col = _find_position_column(df.columns)
        trait_col = _find_column(df.columns, {"trait", "traits", "phenotype", "phenotypes"})
        p_col = _find_pvalue_column(df.columns)
        if p_col is None or (marker_col is None and (chrom_col is None or pos_col is None)):
            continue
        files_used.append(str(path))
        inferred_trait_name, inferred_trait_idx = _infer_trait_from_path(path, trait_names, trait_lookup)
        for _, row in df.iterrows():
            marker_idx = _marker_index_from_row(
                row=row,
                marker_col=marker_col,
                chrom_col=chrom_col,
                pos_col=pos_col,
                marker_lookup=marker_lookup,
                locus_lookup=locus_lookup,
                marker_count=len(marker_names),
            )
            if marker_idx is None:
                continue
            if trait_col is not None:
                trait_name, trait_idx = _normalize_trait_name(row[trait_col], trait_lookup)
                if trait_idx is None:
                    continue
                if trait_name is None:
                    trait_name = trait_names[trait_idx]
            elif inferred_trait_idx is not None:
                trait_name = inferred_trait_name or trait_names[inferred_trait_idx]
                trait_idx = inferred_trait_idx
            elif len(trait_names) == 1:
                trait_name = trait_names[0]
                trait_idx = 0
            else:
                continue
            p_float = _parse_pvalue(row[p_col])
            if p_float is None:
                continue
            score = float(-np.log10(max(p_float, np.finfo(float).tiny)))
            if score > raw_scores[trait_idx, marker_idx]:
                raw_scores[trait_idx, marker_idx] = score
                raw_pvalues[trait_idx, marker_idx] = p_float
            parsed_rows.append(
                {
                    "marker": marker_names[marker_idx],
                    "trait": trait_name,
                    "pvalue": p_float,
                    "score": score,
                    "source_file": str(path),
                }
            )

    if not parsed_rows:
        raise ValueError(
            "TASSEL MLM finished but no marker p-values could be parsed. "
            "Please inspect the mlm_output files saved under the tassel_mlm_gwas directory. "
            f"Inspected files: {files_inspected[:5]}"
        )

    prior_table = pd.DataFrame({"marker": marker_names})
    for trait_idx, trait in enumerate(trait_names):
        prior_table[trait] = raw_scores[trait_idx]
        prior_table[f"{trait}_pvalue"] = raw_pvalues[trait_idx]

    parsed_df = pd.DataFrame(parsed_rows)
    summary_traits: dict[str, object] = {}
    bonferroni_threshold = 0.05 / max(len(marker_names), 1)
    for trait_idx, trait in enumerate(trait_names):
        trait_scores = raw_scores[trait_idx]
        trait_p = raw_pvalues[trait_idx]
        positive = trait_scores > 0
        top_idx = np.argsort(trait_scores)[::-1][:50]
        summary_traits[trait] = {
            "parsed_markers": int(np.sum(positive)),
            "best_score": float(np.nanmax(trait_scores)) if np.any(positive) else None,
            "best_pvalue": float(np.nanmin(trait_p)) if np.any(np.isfinite(trait_p)) else None,
            "bonferroni_threshold": float(bonferroni_threshold),
            "bonferroni_significant": int(np.sum(np.isfinite(trait_p) & (trait_p <= bonferroni_threshold))),
            "top_markers": [
                {
                    "marker": marker_names[int(idx)],
                    "score": float(trait_scores[int(idx)]),
                    "pvalue": float(trait_p[int(idx)]) if np.isfinite(trait_p[int(idx)]) else None,
                }
                for idx in top_idx
                if trait_scores[int(idx)] > 0
            ],
        }

    parse_summary = {
        "files_used": files_used,
        "parsed_rows": int(len(parsed_rows)),
        "traits": summary_traits,
    }
    return raw_scores, prior_table, parse_summary


def _soft_prior_scale(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    scaled = np.zeros_like(scores, dtype=np.float32)
    for trait_idx in range(scores.shape[0]):
        row = scores[trait_idx]
        positive = row[row > 0]
        if positive.size > 1:
            mean = float(positive.mean())
            std = float(positive.std())
            if std < 1e-8:
                scaled[trait_idx, row > 0] = 1.0
            else:
                scaled_values = (positive - mean) / std + 1.0
                scaled[trait_idx, row > 0] = np.clip(scaled_values, 0.0, 5.0)
        elif positive.size == 1:
            scaled[trait_idx, row > 0] = 1.0
    return scaled


def build_tassel_mlm_prior(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    y_mask: np.ndarray,
    sample_ids: list[str],
    marker_names: list[str],
    trait_names: list[str],
    output_dir: Path,
    tassel_pipeline_path: str | Path | None = None,
    pc_count: int = 3,
    max_p: float | None = None,
    prior_sparsity: str | None = "top_1pct",
    force_synthetic_positions: bool = False,
) -> TasselPriorResult:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    safe_sample_ids = _unique_safe_names(sample_ids)
    hapmap_path = input_dir / "genotype_for_tassel.hmp.txt"
    phenotype_path = input_dir / "phenotype_for_tassel.txt"
    kinship_path = input_dir / "kinship_for_tassel.txt"
    mlm_output_prefix = output_dir / "mlm_output.txt"

    hapmap_summary = write_tassel_hapmap(
        x_raw,
        safe_sample_ids,
        marker_names,
        hapmap_path,
        force_synthetic_positions=force_synthetic_positions,
    )
    phenotype_summary = write_tassel_phenotype(
        y_raw,
        y_mask,
        safe_sample_ids,
        trait_names,
        x_raw,
        phenotype_path,
        pc_count=pc_count,
    )
    kinship_summary = write_tassel_kinship(x_raw, safe_sample_ids, kinship_path)
    (
        tassel_hapmap_path,
        tassel_phenotype_path,
        tassel_kinship_path,
        tassel_mlm_output_prefix,
        tassel_runtime_dir,
    ) = _prepare_tassel_runtime_paths(hapmap_path, phenotype_path, kinship_path, mlm_output_prefix)

    pipeline_path = resolve_tassel_pipeline(tassel_pipeline_path)
    command_text: str | None = None
    command_script = output_dir / "run_tassel_mlm.bat"
    if pipeline_path is not None:
        command = _build_tassel_command(
            pipeline_path,
            tassel_hapmap_path,
            tassel_phenotype_path,
            tassel_kinship_path,
            tassel_mlm_output_prefix,
            max_p=max_p,
        )
        command_text = " ".join(_quote_for_bat(part) for part in command)
        tassel_working_dir = pipeline_path.parent
        _write_command_script(command, command_script, working_dir=tassel_working_dir)
        process = subprocess.run(
            command,
            cwd=str(tassel_working_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=None,
        )
        (output_dir / "tassel_stdout.log").write_text(process.stdout or "", encoding="utf-8")
        (output_dir / "tassel_stderr.log").write_text(process.stderr or "", encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                "TASSEL 5 MLM GWAS failed. "
                f"Command script: {command_script}. "
                f"stderr log: {output_dir / 'tassel_stderr.log'}"
            )
        runtime_outputs = _copy_tassel_runtime_outputs(tassel_runtime_dir, output_dir)
    else:
        runtime_outputs = []
        fallback_command = _build_tassel_command(
            Path("run_pipeline.bat"),
            hapmap_path,
            phenotype_path,
            kinship_path,
            mlm_output_prefix,
            max_p=max_p,
        )
        command_text = " ".join(_quote_for_bat(part) for part in fallback_command)
        _write_command_script(fallback_command, command_script)
        summary = {
            "enabled": True,
            "method": "tassel_mlm_gwas_prior",
            "status": "not_run",
            "reason": "TASSEL 5 run_pipeline.bat was not found.",
            "download_url": TASSEL_DOWNLOAD_URL,
            "documentation_url": TASSEL_DOC_URL,
            "command_script": str(command_script),
            "inputs": {
                "hapmap": hapmap_summary,
                "phenotype": phenotype_summary,
                "kinship": kinship_summary,
            },
            "command": command_text,
        }
        (output_dir / "tassel_prior_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        raise RuntimeError(
            "TASSEL 5 run_pipeline.bat was not found. "
            f"I generated TASSEL input files and a command script here: {output_dir}. "
            f"Install TASSEL 5 from {TASSEL_DOWNLOAD_URL}, install Java 1.8+, "
            "then set the TASSEL run_pipeline.bat path in the UI or TASSEL5_RUN_PIPELINE environment variable."
        )

    output_paths = _candidate_tassel_outputs(output_dir, mlm_output_prefix)
    raw_scores, prior_table, parse_summary = _parse_tassel_outputs(output_paths, marker_names, trait_names)
    sparse_raw_scores, sparsity_summary = _apply_prior_sparsity(raw_scores, prior_sparsity)
    for trait_idx, trait in enumerate(trait_names):
        prior_table[trait] = sparse_raw_scores[trait_idx]
        prior_table[f"{trait}_raw_score_before_sparsity"] = raw_scores[trait_idx]
    scaled_scores = _soft_prior_scale(sparse_raw_scores)

    prior_path = output_dir / "snp_marker_tassel_mlm_prior.csv"
    prior_table.to_csv(prior_path, index=False)
    raw_hits_path = output_dir / "tassel_mlm_marker_hits_long.csv"
    long_rows = []
    for trait_idx, trait in enumerate(trait_names):
        for marker_idx, marker in enumerate(marker_names):
            score = float(sparse_raw_scores[trait_idx, marker_idx])
            if score <= 0:
                continue
            pvalue = prior_table.loc[marker_idx, f"{trait}_pvalue"]
            long_rows.append(
                {
                    "marker": marker,
                    "trait": trait,
                    "score": score,
                    "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
                }
            )
    pd.DataFrame(long_rows).to_csv(raw_hits_path, index=False)

    summary = {
        "enabled": True,
        "method": "tassel_mlm_gwas_prior",
        "engine": "TASSEL 5 MLM",
        "status": "completed",
        "score_definition": "-log10(TASSEL_MLM_pvalue)",
        "score_transform": "positive_scores_standardized_per_trait_to_soft_prior",
        "prior_sparsity": sparsity_summary,
        "tassel_pipeline_path": str(pipeline_path),
        "pc_covariate_count": int(pc_count),
        "inputs": {
            "hapmap": hapmap_summary,
            "phenotype": phenotype_summary,
            "kinship": kinship_summary,
        },
        "command": command_text,
        "command_script": str(command_script),
        "runtime_dir": str(tassel_runtime_dir) if tassel_runtime_dir is not None else None,
        "runtime_outputs_copied": runtime_outputs,
        "prior_file": str(prior_path),
        "raw_hits_file": str(raw_hits_path),
        "parse": parse_summary,
    }
    (output_dir / "tassel_prior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return TasselPriorResult(
        prior_scores=scaled_scores.astype(np.float32),
        summary=summary,
        prior_table=prior_table,
    )
