from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import training  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute expected-gradient attributions for a saved PPMGS-Net "
            "checkpoint and export per-sample values for SHAP beeswarm plots."
        )
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--genotype", required=True, type=Path)
    parser.add_argument("--phenotype", required=True, type=Path)
    parser.add_argument("--ranking", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gradient-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="User-selected seed for expected-gradient background sampling.",
    )
    return parser.parse_args()


def _sample_id_column(frame: pd.DataFrame) -> str:
    candidates = ["sample_id", "individual_id", "id", "ID"]
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0])


def _prepare_inputs(
    job: training.TrainedJob,
    genotype_path: Path,
    phenotype_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    phenotype = pd.read_csv(phenotype_path)
    phenotype_id = _sample_id_column(phenotype)
    missing_traits = [trait for trait in job.trait_names if trait not in phenotype.columns]
    if missing_traits:
        raise ValueError(f"Phenotype file is missing traits: {missing_traits}")

    phenotype = phenotype[[phenotype_id, *job.trait_names]].copy()
    phenotype[job.trait_names] = phenotype[job.trait_names].apply(
        pd.to_numeric, errors="coerce"
    )
    phenotype = phenotype.loc[
        phenotype[job.trait_names].notna().any(axis=1)
    ].copy()

    header = pd.read_csv(genotype_path, nrows=0)
    genotype_id = _sample_id_column(header)
    missing_markers = [
        marker for marker in job.marker_names if marker not in header.columns
    ]
    if missing_markers:
        raise ValueError(
            f"Genotype file is missing {len(missing_markers)} model markers; "
            f"first five: {missing_markers[:5]}"
        )

    genotype = pd.read_csv(
        genotype_path,
        usecols=[genotype_id, *job.marker_names],
    )
    # Match prepare_training_data(): genotype is the left table, so its row
    # order is preserved for the deterministic expected-gradient background draws.
    aligned = genotype.merge(
        phenotype[[phenotype_id]],
        left_on=genotype_id,
        right_on=phenotype_id,
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != job.samples:
        raise ValueError(
            f"Aligned sample count {len(aligned)} does not match saved model "
            f"sample count {job.samples}."
        )

    raw = aligned[job.marker_names].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32,
        copy=True,
    )
    fill_values = np.asarray(job.marker_fill_values, dtype=np.float32)
    missing = ~np.isfinite(raw)
    if missing.any():
        raw[missing] = np.broadcast_to(fill_values, raw.shape)[missing]

    x_std = np.asarray(job.x_std, dtype=np.float32)
    x_std = np.where(np.abs(x_std) < 1e-12, 1.0, x_std)
    standardized = (raw - np.asarray(job.x_mean, dtype=np.float32)) / x_std
    sample_ids = aligned[genotype_id].astype(str).tolist()
    return raw, standardized.astype(np.float32, copy=False), sample_ids


def _load_ranking(
    job: training.TrainedJob,
    ranking_path: Path,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    ranking = pd.read_csv(ranking_path)
    required = {"trait", "rank", "marker", "mean_abs_shap"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"Ranking file is missing columns: {sorted(missing)}")

    marker_lookup = {marker: index for index, marker in enumerate(job.marker_names)}
    output: dict[str, pd.DataFrame] = {}
    for trait in job.trait_names:
        trait_ranking = (
            ranking.loc[ranking["trait"].astype(str) == trait]
            .sort_values("rank")
            .head(top_k)
            .copy()
        )
        if len(trait_ranking) != top_k:
            raise ValueError(
                f"{trait}: expected {top_k} ranked markers, found {len(trait_ranking)}"
            )
        absent = [
            marker
            for marker in trait_ranking["marker"].astype(str)
            if marker not in marker_lookup
        ]
        if absent:
            raise ValueError(f"{trait}: ranked markers absent from model: {absent}")
        trait_ranking["marker_index"] = [
            marker_lookup[marker]
            for marker in trait_ranking["marker"].astype(str)
        ]
        output[trait] = trait_ranking
    return output


def _export_attributions(
    job: training.TrainedJob,
    raw: np.ndarray,
    x: np.ndarray,
    sample_ids: list[str],
    rankings: dict[str, pd.DataFrame],
    out_dir: Path,
    gradient_samples: int,
    batch_size: int,
) -> dict[str, object]:
    model = job.model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The saved job is not a torch PPMGS-Net model.")

    device = next(model.parameters()).device
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    sample_count = x.shape[0]
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {}

    old_training = model.training
    old_attention_scale = getattr(model, "attention_runtime_scale", None)
    model.eval()
    training._set_attention_runtime_scale(model, 1.0)
    try:
        for trait_index, trait in enumerate(job.trait_names):
            ranked = rankings[trait]
            marker_indices = ranked["marker_index"].to_numpy(dtype=np.int64)
            marker_names = ranked["marker"].astype(str).tolist()
            ranks = ranked["rank"].astype(int).tolist()
            saved_mean_abs = ranked["mean_abs_shap"].to_numpy(dtype=float)

            generator = torch.Generator(device=device)
            generator.manual_seed(training._training_seed(7877 + trait_index))
            top_attributions = np.empty(
                (sample_count, len(marker_indices)), dtype=np.float32
            )

            for start in range(0, sample_count, batch_size):
                stop = min(start + batch_size, sample_count)
                x_batch = x_tensor[start:stop]
                batch_n = x_batch.shape[0]
                attribution_sum = torch.zeros_like(x_batch)
                for _ in range(gradient_samples):
                    background_index = torch.randint(
                        0,
                        sample_count,
                        (batch_n,),
                        device=device,
                        generator=generator,
                    )
                    background = x_tensor[background_index]
                    alpha = torch.rand(
                        (batch_n, 1),
                        device=device,
                        generator=generator,
                    )
                    interpolated = (
                        background + alpha * (x_batch - background)
                    ).detach().requires_grad_(True)
                    model.zero_grad(set_to_none=True)
                    output = model(interpolated)[:, trait_index].sum()
                    gradient = torch.autograd.grad(
                        output,
                        interpolated,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    attribution_sum += (x_batch - background) * gradient

                attribution = (
                    attribution_sum / float(gradient_samples)
                ).detach().cpu().numpy()
                top_attributions[start:stop] = attribution[:, marker_indices]

            recomputed_mean_abs = np.mean(np.abs(top_attributions), axis=0)
            relative_error = np.abs(recomputed_mean_abs - saved_mean_abs) / np.maximum(
                saved_mean_abs, 1e-12
            )
            diagnostics[trait] = {
                "top_k": len(marker_indices),
                "max_relative_difference_from_saved_mean_abs": float(
                    np.max(relative_error)
                ),
                "mean_relative_difference_from_saved_mean_abs": float(
                    np.mean(relative_error)
                ),
            }

            for marker_offset, (rank, marker, marker_index) in enumerate(
                zip(ranks, marker_names, marker_indices, strict=True)
            ):
                for sample_index, sample_id in enumerate(sample_ids):
                    rows.append(
                        {
                            "trait": trait,
                            "rank": rank,
                            "marker": marker,
                            "sample_index": sample_index,
                            "sample_id": sample_id,
                            "shap_value": float(
                                top_attributions[sample_index, marker_offset]
                            ),
                            "genotype_dosage": float(raw[sample_index, marker_index]),
                            "feature_value_standardized": float(
                                x[sample_index, marker_index]
                            ),
                        }
                    )
    finally:
        if old_attention_scale is not None:
            training._set_attention_runtime_scale(
                model, float(old_attention_scale)
            )
        model.train(old_training)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "shap_top20_sample_values.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    return {
        "method": "PPMGS-Net",
        "sample_count": sample_count,
        "trait_count": len(job.trait_names),
        "top_k": len(next(iter(rankings.values()))),
        "row_count": len(rows),
        "gradient_samples": gradient_samples,
        "batch_size": batch_size,
        "output_csv": str(output_csv),
        "diagnostics": diagnostics,
    }


def main() -> None:
    args = _parse_args()
    training.configure_training_seed(args.seed)
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    job = training.load_job(args.job_id)
    job_dir = training.SAVED_MODELS_DIR / args.job_id
    ranking_path = args.ranking or (job_dir / "shap_top50_markers.csv")
    if not ranking_path.exists():
        raise FileNotFoundError(ranking_path)

    raw, standardized, sample_ids = _prepare_inputs(
        job,
        args.genotype,
        args.phenotype,
    )
    rankings = _load_ranking(job, ranking_path, args.top_k)
    summary = _export_attributions(
        job,
        raw,
        standardized,
        sample_ids,
        rankings,
        args.out_dir,
        max(1, args.gradient_samples),
        max(1, args.batch_size),
    )
    summary["job_id"] = args.job_id
    summary["ranking"] = str(ranking_path)
    summary_path = args.out_dir / "shap_top20_sample_values_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
