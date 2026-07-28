#!/usr/bin/env bash
set -euo pipefail

: "${GENOTYPE:?Set GENOTYPE to a genotype CSV.}"
: "${PHENOTYPE:?Set PHENOTYPE to a phenotype CSV.}"
: "${TRAITS:?Set TRAITS to a comma-separated trait list.}"
: "${TASSEL_PIPELINE:?Set TASSEL_PIPELINE to TASSEL run_pipeline.pl or run_pipeline.bat.}"
: "${SEED:?Choose and set SEED.}"

OUT_DIR="${OUT_DIR:-results/full_model}"

python scripts/run_training_cli.py \
  --genotype "${GENOTYPE}" \
  --phenotype "${PHENOTYPE}" \
  --out-dir "${OUT_DIR}" \
  --model-family ppmgs \
  --task-type multi_trait \
  --trait-names "${TRAITS}" \
  --allow-missing \
  --attention-mode prior_marker_pearson \
  --attention-blend-metric pearson_learned_alpha \
  --trait-gate-mode directional_anchor \
  --pdae \
  --tassel-prior \
  --tassel-pipeline "${TASSEL_PIPELINE}" \
  --lasso-prior \
  --prior-sparsity top_1pct \
  --cv \
  --cv-folds 5 \
  --cv-repeats 10 \
  --save-oof-intervals \
  --shap \
  --seed "${SEED}"
