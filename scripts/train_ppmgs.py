from __future__ import annotations

"""Clean command-line entry point for the final PPMGS-Net workflow.

This module exposes the architecture used in the manuscript: trait-specific
prior-aware anchors, frozen-anchor directional transfer, and the optional
phenotype-missing-aware PDAE teacher.
"""

import argparse
import json
import shutil
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.training_ppmgs_final import (  # noqa: E402
    SAVED_MODELS_DIR,
    configure_training_seed,
    train_model,
)


FINAL_ATTENTION_MODES = ("none", "prior_marker_pearson")
FINAL_TRANSFER_MODES = ("none", "directional_anchor")


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _load_ppmgs_params(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--ppmgs-params-json must contain a JSON object.")

    candidates = [payload.get("hyperparameters"), payload.get("best_params")]
    search = payload.get("hyperparameter_search")
    if isinstance(search, dict):
        candidates.append(search.get("best_params"))
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return dict(candidate)

    known_keys = {
        "hidden_dim",
        "hidden_layers",
        "dropout",
        "activation",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "pdae_mask_rate",
        "pdae_loss_weight",
        "pdae_pseudo_weight",
        "pdae_hidden_dim",
        "lasso_prior_gwas_weight",
    }
    if any(key in payload for key in known_keys):
        return dict(payload)
    raise ValueError(
        "The parameter file must contain hyperparameters, best_params, "
        "hyperparameter_search.best_params, or a bare PPMGS parameter object."
    )


def _resolve_transfer_mode(task_type: str, requested: str | None) -> str:
    if requested is not None:
        return requested
    return "directional_anchor" if task_type == "multi_trait" else "none"


def _validate_args(args: argparse.Namespace, transfer_mode: str) -> None:
    for path in (args.genotype, args.phenotype):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.phenotype_truth, args.prior_marker, args.ppmgs_params_json):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    if args.task_type == "single_trait":
        if not args.trait_name:
            raise ValueError("--trait-name is required for single-trait training.")
        if args.trait_names:
            raise ValueError("Use --trait-name, not --trait-names, for single-trait training.")
        if transfer_mode != "none":
            raise ValueError("Single-trait training does not use directional transfer.")
    else:
        trait_names = _csv_list(args.trait_names)
        if trait_names is None or len(trait_names) < 2:
            raise ValueError("--trait-names must contain at least two comma-separated traits.")
        if args.trait_name:
            raise ValueError("Use --trait-names, not --trait-name, for multi-trait training.")

    if transfer_mode == "directional_anchor" and args.attention_mode != "prior_marker_pearson":
        raise ValueError("directional_anchor requires --attention-mode prior_marker_pearson.")
    if args.pdae and args.task_type != "multi_trait":
        raise ValueError("PDAE is available only for multi-trait training.")
    if args.hyperopt and args.ppmgs_params_json is not None:
        raise ValueError("--hyperopt and --ppmgs-params-json cannot be used together.")
    if args.profile_resources and not torch.cuda.is_available():
        raise RuntimeError("--profile-resources requires an available CUDA device.")


def _compact_result(
    job_id: str,
    job,
    observed: int,
    seed: int,
    elapsed_seconds: float,
    resource_profile: dict[str, object] | None,
) -> dict[str, object]:
    pdae_summary = job.pdae_summary if isinstance(job.pdae_summary, dict) else {}
    result = {
        "job_id": job_id,
        "saved_model_dir": str(SAVED_MODELS_DIR / job_id),
        "method": "PPMGS-Net",
        "task_type": job.task_type,
        "transfer_mode": job.transfer_mode,
        "attention_mode": job.attention_mode,
        "attention_architecture": getattr(job.model, "attention_architecture", None),
        "pdae_enabled": bool(pdae_summary.get("enabled", False)),
        "samples": job.samples,
        "markers": len(job.marker_names),
        "traits": job.trait_names,
        "observed_trait_values": observed,
        "random_seed": int(seed),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "hyperparameters": job.hyperparameters,
        "hyperparameter_search": job.hyperparameter_search,
        "prior_marker_summary": job.prior_marker_summary,
        "phenotype_missing_summary": job.phenotype_missing_summary,
        "pdae_summary": job.pdae_summary,
        "directional_transfer_summary": job.directional_transfer_summary,
        "best_epoch": job.best_epoch,
        "final_loss": job.final_loss,
        "validation_metrics": job.metrics,
        "cross_validation": job.cross_validation,
        "prediction_interval_calibration": job.prediction_interval_calibration,
        "uncertainty_metadata": job.uncertainty_metadata,
        "shap_top_markers": job.shap_top_markers,
        "shap_summary": job.shap_summary,
        "timing_summary": job.timing_summary,
    }
    if resource_profile is not None:
        result["resource_profile"] = resource_profile
    return result


def _copy_oof_outputs(payload: dict[str, object], target_dir: Path) -> None:
    cross_validation = payload.get("cross_validation")
    if not isinstance(cross_validation, dict):
        return
    intervals = cross_validation.get("oof_prediction_intervals")
    if not isinstance(intervals, dict):
        return
    files = intervals.get("files")
    if not isinstance(files, dict):
        return

    copied: dict[str, str] = {}
    for key, value in files.items():
        if not str(key).endswith("_csv"):
            continue
        source = Path(str(value))
        if not source.exists():
            continue
        destination = target_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        copied[str(key)] = str(destination)
    if copied:
        intervals["copied_files"] = copied


def _write_outputs(payload: dict[str, object], output_dir: Path | None) -> None:
    job_id = str(payload["job_id"])
    target_dir = output_dir or (SAVED_MODELS_DIR / job_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    _copy_oof_outputs(payload, target_dir)

    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    json_path = target_dir / f"{job_id}_training_result.json"
    json_path.write_text(text, encoding="utf-8")

    metadata = SAVED_MODELS_DIR / job_id / "metadata.json"
    if metadata.exists() and metadata.parent.resolve() != target_dir.resolve():
        shutil.copy2(metadata, target_dir / f"{job_id}_metadata_full.json")

    print(text)
    print(f"\nSaved result: {json_path}")
    print(f"Saved model: {SAVED_MODELS_DIR / job_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the final PPMGS-Net architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genotype", required=True, type=Path)
    parser.add_argument("--phenotype", required=True, type=Path)
    parser.add_argument("--phenotype-truth", type=Path, default=None)
    parser.add_argument("--prior-marker", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)

    parser.add_argument("--task-type", choices=("single_trait", "multi_trait"), required=True)
    parser.add_argument("--trait-name", default=None)
    parser.add_argument("--trait-names", default=None)
    parser.add_argument("--allow-missing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attention-mode", choices=FINAL_ATTENTION_MODES, default="prior_marker_pearson")
    parser.add_argument("--transfer-mode", choices=FINAL_TRANSFER_MODES, default=None)
    parser.add_argument("--pdae", action="store_true")

    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--activation", choices=("relu", "gelu"), default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau", "cosine"), default="plateau")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument("--save-oof-intervals", action="store_true")

    parser.add_argument("--pdae-mask-rate", type=float, default=0.3)
    parser.add_argument("--pdae-loss-weight", type=float, default=0.15)
    parser.add_argument("--pdae-pseudo-weight", type=float, default=0.01)

    parser.add_argument("--tassel-prior", action="store_true")
    parser.add_argument("--tassel-pipeline", default=None)
    parser.add_argument("--tassel-pc-count", type=int, default=3)
    parser.add_argument("--lasso-prior", action="store_true")
    parser.add_argument("--lasso-gwas-weight", type=float, default=0.5)
    parser.add_argument("--lasso-repeats", type=int, default=50)
    parser.add_argument("--lasso-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--prior-sparsity",
        choices=("none", "top_0_5pct", "top_1pct", "top_5pct", "p_1e-3", "p_1e-4"),
        default="top_1pct",
    )

    parser.add_argument("--hyperopt", action="store_true")
    parser.add_argument("--hyperopt-trials", type=int, default=50)
    parser.add_argument("--hyperopt-folds", type=int, default=2)
    parser.add_argument(
        "--hyperopt-metric",
        choices=("pearson", "stable_pearson", "mse", "rmse", "mae"),
        default="pearson",
    )
    parser.add_argument("--hyperopt-early-stop-rounds", type=int, default=20)
    parser.add_argument("--hyperopt-epochs", type=int, default=0)
    parser.add_argument("--ppmgs-params-json", type=Path, default=None)

    parser.add_argument("--shap", action="store_true")
    parser.add_argument("--shap-all-markers", action="store_true")
    parser.add_argument("--shap-top-k", type=int, default=50)
    parser.add_argument("--profile-resources", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    transfer_mode = _resolve_transfer_mode(args.task_type, args.transfer_mode)
    _validate_args(args, transfer_mode)
    configure_training_seed(args.seed)

    params_override = _load_ppmgs_params(args.ppmgs_params_json)
    if args.activation is not None and not args.hyperopt:
        params_override = {**(params_override or {}), "activation": args.activation}

    if args.profile_resources:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started_at = time.perf_counter()
    with ExitStack() as stack:
        genotype_file = stack.enter_context(args.genotype.open("rb"))
        phenotype_file = stack.enter_context(args.phenotype.open("rb"))
        truth_file = (
            stack.enter_context(args.phenotype_truth.open("rb"))
            if args.phenotype_truth is not None
            else None
        )
        prior_file = (
            stack.enter_context(args.prior_marker.open("rb"))
            if args.prior_marker is not None
            else None
        )

        job_id, job, observed = train_model(
            genotype_file,
            phenotype_file,
            prior_marker_file=prior_file,
            task_type=args.task_type,
            trait_name=args.trait_name,
            trait_names=_csv_list(args.trait_names),
            allow_missing_phenotype=args.allow_missing,
            attention_mode=args.attention_mode,
            use_directional_transfer=transfer_mode == "directional_anchor",
            transfer_mode=transfer_mode,
            use_pdae=args.pdae,
            pdae_mask_rate=args.pdae_mask_rate,
            pdae_loss_weight=args.pdae_loss_weight,
            pdae_pseudo_weight=args.pdae_pseudo_weight,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            use_cross_validation=args.cv,
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            save_oof_intervals=args.save_oof_intervals,
            build_tassel_prior=args.tassel_prior,
            tassel_pipeline_path=args.tassel_pipeline,
            tassel_pc_count=args.tassel_pc_count,
            build_lasso_prior=args.lasso_prior,
            lasso_prior_gwas_weight=args.lasso_gwas_weight,
            lasso_prior_repeats=args.lasso_repeats,
            lasso_prior_cache_dir=args.lasso_cache_dir,
            prior_sparsity=args.prior_sparsity,
            optimize_hyperparameters=args.hyperopt,
            hyperparameter_trials=args.hyperopt_trials,
            hyperparameter_folds=args.hyperopt_folds,
            hyperparameter_metric=args.hyperopt_metric,
            hyperparameter_method="tpe",
            hyperparameter_early_stop_rounds=args.hyperopt_early_stop_rounds,
            hyperparameter_max_epochs=args.hyperopt_epochs,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            min_learning_rate=args.min_lr,
            ppmgs_params_override=params_override,
            compute_shap=bool(args.shap or args.shap_all_markers),
            shap_top_k=args.shap_top_k,
            save_full_shap=args.shap_all_markers,
            phenotype_truth_file=truth_file,
        )

    elapsed_seconds = time.perf_counter() - started_at
    resource_profile = None
    if args.profile_resources:
        torch.cuda.synchronize()
        total_parameters = sum(parameter.numel() for parameter in job.model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in job.model.parameters() if parameter.requires_grad
        )
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        resource_profile = {
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "total_parameters": int(total_parameters),
            "trainable_parameters_at_return": int(trainable_parameters),
            "peak_cuda_memory_allocated_mb": round(peak_allocated / 1024**2, 3),
            "peak_cuda_memory_reserved_mb": round(peak_reserved / 1024**2, 3),
            "includes_hyperparameter_search": bool(args.hyperopt),
        }

    payload = _compact_result(
        job_id,
        job,
        observed,
        args.seed,
        elapsed_seconds,
        resource_profile,
    )
    _write_outputs(payload, args.out_dir)


if __name__ == "__main__":
    main()
