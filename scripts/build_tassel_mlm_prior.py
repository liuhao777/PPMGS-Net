from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.gwas_prior import build_tassel_mlm_prior  # noqa: E402
from backend.app.training import prepare_training_data  # noqa: E402


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TASSEL-MLM GWAS SNP-Marker prior from PPMGS-Net genotype/phenotype CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genotype", required=True, type=Path, help="Input genotype.csv with sample_id column.")
    parser.add_argument("--phenotype", required=True, type=Path, help="Input phenotype.csv with sample_id column.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for TASSEL inputs and prior files.")
    parser.add_argument("--task-type", choices=["single_trait", "multi_trait"], default="multi_trait")
    parser.add_argument("--trait-name", default=None, help="Single-trait name.")
    parser.add_argument("--trait-names", default=None, help="Comma-separated multi-trait names.")
    parser.add_argument("--allow-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--tassel-pipeline",
        default=None,
        help="Path to TASSEL 5 run_pipeline.pl or run_pipeline.bat.",
    )
    parser.add_argument("--pc-count", type=int, default=3, help="Number of genotype PCs used as covariates.")
    parser.add_argument(
        "--prior-sparsity",
        choices=["none", "top_0_5pct", "top_1pct", "top_5pct", "p_1e-3", "p_1e-4"],
        default="top_1pct",
        help="Keep only selected GWAS prior markers; all other SNP prior scores become 0.",
    )
    args = parser.parse_args()

    with ExitStack() as stack:
        genotype_file = stack.enter_context(args.genotype.open("rb"))
        phenotype_file = stack.enter_context(args.phenotype.open("rb"))
        (
            x,
            y,
            mask,
            marker_names,
            trait_names,
            x_mean,
            x_std,
            y_mean,
            y_std,
            _marker_fill_values,
            phenotype_missing_summary,
            sample_ids,
        ) = prepare_training_data(
            genotype_file,
            phenotype_file,
            task_type=args.task_type,
            allow_missing_phenotype=args.allow_missing,
            trait_name=args.trait_name,
            trait_names=_csv_list(args.trait_names),
        )

    x_raw = x * x_std + x_mean
    y_raw = y * y_std + y_mean
    y_raw = np.where(mask > 0, y_raw, np.nan)
    result = build_tassel_mlm_prior(
        x_raw=x_raw,
        y_raw=y_raw,
        y_mask=mask,
        sample_ids=sample_ids,
        marker_names=marker_names,
        trait_names=trait_names,
        output_dir=args.out_dir,
        tassel_pipeline_path=args.tassel_pipeline,
        pc_count=args.pc_count,
        prior_sparsity=args.prior_sparsity,
    )
    payload = {
        "prior_file": result.summary.get("prior_file"),
        "summary": result.summary,
        "phenotype_missing_summary": phenotype_missing_summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
