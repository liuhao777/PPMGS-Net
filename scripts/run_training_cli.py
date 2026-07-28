from __future__ import annotations

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

from backend.app.training import (  # noqa: E402
    PUBLIC_ATTENTION_MODES,
    SAVED_MODELS_DIR,
    configure_training_seed,
    train_model,
)


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _load_ppmgs_params(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--ppmgs-params-json must contain a JSON object.")

    candidates = [
        payload.get("hyperparameters"),
        payload.get("best_params"),
    ]
    hyperparameter_search = payload.get("hyperparameter_search")
    if isinstance(hyperparameter_search, dict):
        candidates.append(hyperparameter_search.get("best_params"))

    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return dict(candidate)

    known_keys = {
        "hidden_dim",
        "hidden_layers",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "pdae_mask_rate",
        "pdae_loss_weight",
        "pdae_pseudo_weight",
        "pdae_hidden_dim",
        "block_size",
    }
    if any(key in payload for key in known_keys):
        return dict(payload)

    raise ValueError(
        "--ppmgs-params-json must contain hyperparameters, best_params, "
        "hyperparameter_search.best_params, or a bare PPMGS-Net params object."
    )


def _compact_result(
    job_id: str,
    job,
    observed: int,
    random_seed: int,
    elapsed_seconds: float | None = None,
    resource_profile: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = {
        "job_id": job_id,
        "saved_model_dir": str(SAVED_MODELS_DIR / job_id),
        "model_family": job.model_family,
        "task_type": job.task_type,
        "allow_missing_phenotype": job.allow_missing_phenotype,
        "use_trait_gate": job.use_trait_gate,
        "trait_gate_mode": job.trait_gate_mode,
        "attention_mode": job.attention_mode,
        "attention_architecture": getattr(job.model, "attention_architecture", None),
        "samples": job.samples,
        "markers": len(job.marker_names),
        "traits": job.trait_names,
        "observed_trait_values": observed,
        "random_seed": int(random_seed),
        "genotype_imputation": job.imputation_strategy,
        "hyperparameters": job.hyperparameters,
        "hyperparameter_search": job.hyperparameter_search,
        "prior_marker_summary": job.prior_marker_summary,
        "pdae_summary": job.pdae_summary,
        "phenotype_missing_summary": job.phenotype_missing_summary,
        "attention_safety": job.attention_safety,
        "trait_interaction_summary": job.trait_interaction_summary,
        "best_epoch": job.best_epoch,
        "final_loss": job.final_loss,
        "validation_metrics": job.metrics,
        "cross_validation": job.cross_validation,
        "conformal_prediction": job.conformal_prediction,
        "conformal_coverage": job.conformal_coverage,
        "individualized_conformal_prediction": getattr(job, "individualized_conformal_prediction", None),
        "uncertainty_metadata": job.uncertainty_metadata,
        "shap_top_markers": job.shap_top_markers,
        "shap_summary": job.shap_summary,
        "timing_summary": job.timing_summary,
    }
    if resource_profile is not None:
        payload["resource_profile"] = resource_profile
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = round(float(elapsed_seconds), 2)
        payload["elapsed_minutes"] = round(float(elapsed_seconds) / 60.0, 4)
    return payload


def _write_outputs(payload: dict[str, object], out_dir: Path | None) -> None:
    job_id = str(payload["job_id"])
    target_dir = out_dir or (SAVED_MODELS_DIR / job_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    cross_validation = payload.get("cross_validation")
    if isinstance(cross_validation, dict):
        oof_intervals = cross_validation.get("oof_prediction_intervals")
        if isinstance(oof_intervals, dict):
            files = oof_intervals.get("files")
            copied_files: dict[str, str] = {}
            if isinstance(files, dict):
                for key, value in files.items():
                    if not str(key).endswith("_csv"):
                        continue
                    source = Path(str(value))
                    if not source.exists():
                        continue
                    destination = target_dir / source.name
                    if source.resolve() != destination.resolve():
                        shutil.copy2(source, destination)
                    copied_files[str(key)] = str(destination)
            if copied_files:
                oof_intervals["copied_files"] = copied_files

    json_path = target_dir / f"{job_id}_training_result.json"
    txt_path = target_dir / f"{job_id}_training_result.txt"

    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    json_path.write_text(text, encoding="utf-8")
    txt_path.write_text(text + "\n", encoding="utf-8")

    source_metadata = SAVED_MODELS_DIR / job_id / "metadata.json"
    if source_metadata.exists() and source_metadata.parent != target_dir:
        shutil.copy2(source_metadata, target_dir / f"{job_id}_metadata_full.json")

    print(text)
    print(f"\nSaved compact JSON: {json_path}")
    print(f"Saved TXT: {txt_path}")
    print(f"Saved model dir: {SAVED_MODELS_DIR / job_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PPMGS-Net training from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genotype", required=True, type=Path, help="Input genotype.csv with sample_id column.")
    parser.add_argument("--phenotype", required=True, type=Path, help="Input phenotype.csv with sample_id column.")
    parser.add_argument(
        "--phenotype-truth",
        type=Path,
        default=None,
        help="Optional complete phenotype CSV used only to evaluate artificially hidden cells; never used in training loss.",
    )
    parser.add_argument("--prior-marker", type=Path, default=None, help="Optional SNP-Marker prior CSV.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional directory for compact result JSON/TXT.")

    parser.add_argument(
        "--model-family",
        default="ppmgs",
        help="Use ppmgs for PPMGS-Net. Baselines include deepgp_st_mlp, deepgp_mt_mlp, mnndr_st, mnndr_mt, ridge, gblup, mt_gblup, mt_bglr, random_forest, xgboost, svm, cnn, etc.",
    )
    parser.add_argument("--task-type", choices=["single_trait", "multi_trait"], default="multi_trait")
    parser.add_argument("--trait-name", default=None, help="Single-trait name.")
    parser.add_argument("--trait-names", default=None, help="Comma-separated multi-trait names, e.g. weight,death1.")
    parser.add_argument("--allow-missing", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument(
        "--attention-mode",
        choices=sorted(PUBLIC_ATTENTION_MODES),
        default="prior_marker_pearson",
    )
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Base random seed for CV splits, model initialization, and data-loader shuffling.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--activation", choices=["relu", "gelu"], default=None, help="Fixed PPMGS-Net activation when hyperopt is off.")
    parser.add_argument("--lr-scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=10)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--cv", action="store_true", help="Enable repeated K-fold cross-validation.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument(
        "--save-oof-intervals",
        action="store_true",
        help="During CV, save ordinary repeated K-fold OOF 95%% interval CSV files using the same validation-fold predictions as the main CV metrics.",
    )

    parser.add_argument(
        "--pdae",
        action="store_true",
        help="Enable fold-local PDAE pretraining and confidence-filtered frozen-teacher pseudo labels.",
    )
    parser.add_argument(
        "--trait-gate-mode",
        choices=("none", "directional_anchor"),
        default=None,
        help=(
            "Use directional_anchor for the final PPMGS-Net multi-trait transfer "
            "architecture or none for the transfer ablation. The default is "
            "directional_anchor for multi-trait tasks and none for single-trait tasks."
        ),
    )
    parser.add_argument("--pdae-mask-rate", type=float, default=0.3, help="Probability of masking an eligible complete phenotype row during PDAE pretraining.")
    parser.add_argument("--pdae-loss-weight", type=float, default=0.15, help="Observed-cell reconstruction weight during independent PDAE pretraining.")
    parser.add_argument("--pdae-pseudo-weight", type=float, default=0.01, help="Maximum confidence-weighted pseudo-label loss coefficient after warmup.")

    parser.add_argument("--lasso-prior", action="store_true", help="Build LASSO-GWAS prior for prior-aware attention.")
    parser.add_argument("--lasso-gwas-weight", type=float, default=0.5)
    parser.add_argument("--lasso-repeats", type=int, default=50)
    parser.add_argument(
        "--prior-sparsity",
        choices=["none", "top_0_5pct", "top_1pct", "top_5pct", "p_1e-3", "p_1e-4"],
        default="top_1pct",
        help="Keep only selected GWAS/SNP-Marker prior markers; all other SNP prior scores become 0.",
    )
    parser.add_argument(
        "--attention-blend-metric",
        choices=["pearson_learned_alpha"],
        default="pearson_learned_alpha",
        help=(
            "Use Pearson early stopping and the network-learned fusion alpha."
        ),
    )
    parser.add_argument(
        "--tassel-prior",
        action="store_true",
        help=(
            "Build TASSEL-MLM GWAS prior for prior-aware attention. "
            "With cross-validation, TASSEL GWAS and enabled LASSO stability selection "
            "are rebuilt from each training fold only."
        ),
    )
    parser.add_argument(
        "--tassel-pipeline",
        default=None,
        help="Path to TASSEL 5 run_pipeline.pl or run_pipeline.bat.",
    )
    parser.add_argument("--tassel-pc-count", type=int, default=3, help="Number of genotype PCs exported as TASSEL covariates.")

    parser.add_argument("--hyperopt", action="store_true", help="Enable Bayesian/TPE hyperparameter optimization.")
    parser.add_argument("--hyperopt-trials", type=int, default=100)
    parser.add_argument("--hyperopt-folds", type=int, default=3)
    parser.add_argument(
        "--hyperopt-metric",
        choices=["pearson", "stable_pearson", "mse", "rmse", "mae"],
        default="pearson",
    )
    parser.add_argument("--hyperopt-method", choices=["tpe", "random"], default="tpe")
    parser.add_argument("--hyperopt-early-stop-rounds", type=int, default=20)
    parser.add_argument(
        "--hyperopt-epochs",
        type=int,
        default=0,
        help="Optional max epochs per hyperparameter-tuning fold. 0 keeps the backend default.",
    )
    parser.add_argument(
        "--ppmgs-params-json",
        type=Path,
        default=None,
        help="Reuse fixed PPMGS-Net hyperparameters from a previous training result JSON or bare params JSON.",
    )

    parser.add_argument("--shap", action="store_true", help="Compute SHAP-style Top SNP explanations.")
    parser.add_argument("--shap-all-markers", action="store_true", help="Also save all-marker SHAP rows to shap_all_markers.csv.")
    parser.add_argument("--shap-top-k", type=int, default=50)
    parser.add_argument(
        "--profile-resources",
        action="store_true",
        help=(
            "Record model parameter counts and PyTorch CUDA peak allocated/reserved "
            "memory for this run. Resetting the CUDA counters does not alter training."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_training_seed(args.seed)
    trait_gate_mode = args.trait_gate_mode or (
        "directional_anchor" if args.task_type == "multi_trait" else "none"
    )
    if not args.genotype.exists():
        raise FileNotFoundError(args.genotype)
    if not args.phenotype.exists():
        raise FileNotFoundError(args.phenotype)
    if args.phenotype_truth is not None and not args.phenotype_truth.exists():
        raise FileNotFoundError(args.phenotype_truth)
    if args.prior_marker is not None and not args.prior_marker.exists():
        raise FileNotFoundError(args.prior_marker)
    if args.ppmgs_params_json is not None and not args.ppmgs_params_json.exists():
        raise FileNotFoundError(args.ppmgs_params_json)
    if args.hyperopt and args.ppmgs_params_json is not None:
        raise ValueError("--hyperopt and --ppmgs-params-json cannot be used together.")

    if args.profile_resources and not torch.cuda.is_available():
        raise RuntimeError("--profile-resources requires an available CUDA device.")

    train_start = time.perf_counter()
    ppmgs_params_override = _load_ppmgs_params(args.ppmgs_params_json)
    if args.activation is not None and not args.hyperopt:
        if ppmgs_params_override is not None:
            ppmgs_params_override = {**ppmgs_params_override, "activation": args.activation}
        else:
            ppmgs_params_override = {"activation": args.activation}
    if args.profile_resources:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with ExitStack() as stack:
        genotype_file = stack.enter_context(args.genotype.open("rb"))
        phenotype_file = stack.enter_context(args.phenotype.open("rb"))
        phenotype_truth_file = (
            stack.enter_context(args.phenotype_truth.open("rb")) if args.phenotype_truth else None
        )
        prior_marker_file = stack.enter_context(args.prior_marker.open("rb")) if args.prior_marker else None

        job_id, job, observed = train_model(
            genotype_file,
            phenotype_file,
            prior_marker_file=prior_marker_file,
            epochs=args.epochs,
            lr=args.lr,
            model_family=args.model_family,
            task_type=args.task_type,
            allow_missing_phenotype=args.allow_missing,
            use_marker_attention=args.attention_mode != "none",
            attention_mode=args.attention_mode,
            trait_name=args.trait_name,
            trait_names=_csv_list(args.trait_names),
            patience=args.patience,
            use_cross_validation=args.cv,
            cv_folds=args.cv_folds,
            cv_repeats=args.cv_repeats,
            save_oof_intervals=args.save_oof_intervals,
            use_pdae=args.pdae,
            use_trait_gate=trait_gate_mode != "none",
            trait_gate_mode=trait_gate_mode,
            pdae_mask_rate=args.pdae_mask_rate,
            pdae_loss_weight=args.pdae_loss_weight,
            pdae_pseudo_weight=args.pdae_pseudo_weight,
            build_lasso_prior=args.lasso_prior,
            lasso_prior_gwas_weight=args.lasso_gwas_weight,
            lasso_prior_repeats=args.lasso_repeats,
            prior_sparsity=args.prior_sparsity,
            attention_blend_metric=args.attention_blend_metric,
            build_tassel_prior=args.tassel_prior,
            tassel_pipeline_path=args.tassel_pipeline,
            tassel_pc_count=args.tassel_pc_count,
            optimize_hyperparameters=args.hyperopt,
            hyperparameter_trials=args.hyperopt_trials,
            hyperparameter_folds=args.hyperopt_folds,
            hyperparameter_metric=args.hyperopt_metric,
            hyperparameter_method=args.hyperopt_method,
            hyperparameter_early_stop_rounds=args.hyperopt_early_stop_rounds,
            hyperparameter_max_epochs=args.hyperopt_epochs,
            lr_scheduler=args.lr_scheduler,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            min_learning_rate=args.min_lr,
            ppmgs_params_override=ppmgs_params_override,
            compute_shap=bool(args.shap or args.shap_all_markers),
            shap_top_k=args.shap_top_k,
            save_full_shap=args.shap_all_markers,
            phenotype_truth_file=phenotype_truth_file,
        )
    elapsed_seconds = time.perf_counter() - train_start

    resource_profile = None
    if args.profile_resources:
        torch.cuda.synchronize()
        model = job.model
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
        resource_profile = {
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "cuda_device_index": int(torch.cuda.current_device()),
            "pytorch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "batch_size": (job.hyperparameters or {}).get("batch_size"),
            "total_parameters": int(total_parameters),
            "trainable_parameters_at_return": int(trainable_parameters),
            "peak_cuda_memory_allocated_bytes": peak_allocated_bytes,
            "peak_cuda_memory_allocated_mb": round(peak_allocated_bytes / 1024**2, 3),
            "peak_cuda_memory_reserved_bytes": peak_reserved_bytes,
            "peak_cuda_memory_reserved_mb": round(peak_reserved_bytes / 1024**2, 3),
            "measurement_scope": "entire train_model call after CUDA counter reset",
            "includes_hyperparameter_search": bool(args.hyperopt),
            "cv_folds": int(args.cv_folds) if args.cv else 0,
            "cv_repeats": int(args.cv_repeats) if args.cv else 0,
        }
        print(
            "\nRESOURCE PROFILE\n"
            f"  device: {resource_profile['device']}\n"
            f"  parameters: {total_parameters:,}\n"
            f"  trainable at return: {trainable_parameters:,}\n"
            f"  peak allocated: {resource_profile['peak_cuda_memory_allocated_mb']:.3f} MB\n"
            f"  peak reserved: {resource_profile['peak_cuda_memory_reserved_mb']:.3f} MB\n"
            f"  elapsed: {elapsed_seconds:.3f} s",
            flush=True,
        )

    _write_outputs(
        _compact_result(
            job_id,
            job,
            observed,
            random_seed=args.seed,
            elapsed_seconds=elapsed_seconds,
            resource_profile=resource_profile,
        ),
        args.out_dir,
    )


if __name__ == "__main__":
    main()
