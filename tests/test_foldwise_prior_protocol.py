import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import training


def main() -> None:
    training.configure_training_seed(int.from_bytes(os.urandom(4), "little"))
    calls: list[dict[str, object]] = []
    original_tassel = training.build_tassel_mlm_prior

    def fake_tassel_mlm_prior(**kwargs):
        calls.append(
            {
                "sample_ids": list(kwargs["sample_ids"]),
                "y_raw": np.asarray(kwargs["y_raw"]).copy(),
            }
        )
        traits = len(kwargs["trait_names"])
        markers = len(kwargs["marker_names"])
        scores = np.tile(np.linspace(0.1, 1.0, markers, dtype=np.float32), (traits, 1))
        return SimpleNamespace(
            prior_scores=scores,
            summary={"method": "mock_tassel_mlm", "samples": len(kwargs["sample_ids"])},
        )

    training.build_tassel_mlm_prior = fake_tassel_mlm_prior
    try:
        x = np.arange(32, dtype=np.float32).reshape(8, 4)
        y = np.column_stack(
            [
                np.linspace(1.0, 8.0, 8, dtype=np.float32),
                np.linspace(11.0, 18.0, 8, dtype=np.float32),
            ]
        )
        mask = np.ones_like(y, dtype=np.float32)
        builder = training._FoldwisePriorBuilder(
            x=x,
            y=y,
            mask=mask,
            x_mean=np.zeros(4, dtype=np.float32),
            x_std=np.ones(4, dtype=np.float32),
            y_mean=np.zeros(2, dtype=np.float32),
            y_std=np.ones(2, dtype=np.float32),
            sample_ids=[f"sample_{idx}" for idx in range(8)],
            marker_names=[f"marker_{idx}" for idx in range(4)],
            trait_names=["trait_1", "trait_2"],
            output_root=Path(".tmp/foldwise_prior_test_output"),
            build_tassel_prior=True,
            build_lasso_prior=True,
            lasso_gwas_weight=0.5,
            lasso_repeats=2,
            prior_sparsity="none",
            tassel_pipeline_path=None,
            tassel_pc_count=2,
        )

        train_idx = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        validation_idx = np.array([6, 7], dtype=np.int64)
        scores_a, summary_a = builder.build(
            train_idx,
            stage="hyperparameter_tuning",
            repeat_number=1,
            fold_number=1,
            params={"lasso_prior_gwas_weight": 0.25},
        )
        scores_b, _ = builder.build(
            train_idx,
            stage="hyperparameter_tuning",
            repeat_number=1,
            fold_number=1,
            params={"lasso_prior_gwas_weight": 0.75},
        )

        assert len(calls) == 1, "The same tuning fold should reuse cached TASSEL/LASSO components."
        assert calls[0]["sample_ids"] == [f"sample_{idx}" for idx in train_idx]
        assert not any(f"sample_{idx}" in calls[0]["sample_ids"] for idx in validation_idx)
        assert np.array_equal(calls[0]["y_raw"], y[train_idx])
        assert summary_a["scope"] == "training_fold_only"
        assert summary_a["validation_phenotypes_used"] is False
        assert scores_a is not None and scores_b is not None
        assert not np.array_equal(scores_a, scores_b), "TPE weights should re-fuse cached components."

        builder.build(
            np.array([0, 1, 2, 3, 6, 7], dtype=np.int64),
            stage="formal_cross_validation",
            repeat_number=1,
            fold_number=2,
            params={"lasso_prior_gwas_weight": 0.5},
        )
        assert len(calls) == 2, "A different outer fold must rebuild TASSEL/LASSO."
    finally:
        training.build_tassel_mlm_prior = original_tassel

    print("foldwise prior protocol: PASS")


if __name__ == "__main__":
    main()
