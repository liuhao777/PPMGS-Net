# PPMGS-Net

PPMGS-Net is a command-line implementation of prior-informed and
phenotype-missing-aware multi-trait genomic prediction.

The released workflow includes:

- trait-specific GWAS-LASSO marker priors;
- a genome-wide encoder and a trait-specific prior-gated encoder with learned
  path fusion;
- independently trained and frozen trait anchors;
- bounded, low-rank asymmetric directional residual transfer;
- cross-fitted genomic-PDAE teacher predictions with reliability and
  confidence filtering for missing phenotype cells;
- repeated cross-validation, TPE optimization, out-of-fold individualized
  prediction intervals, and expected-gradient SHAP output;
- representative statistical, machine-learning, and deep-learning baselines.

This repository contains the reusable Linux/Python workflow. It intentionally
excludes study datasets, generated priors, saved checkpoints, training logs,
manuscript files, desktop/web interfaces, historical experiments, and
server-specific launch scripts.

## Repository layout

```text
backend/app/
  model.py                  PPMGS-Net neural modules
  training.py               Training, CV, TPE, baselines, OOF and SHAP logic
  gwas_prior.py             TASSEL MLM prior construction
  r_scripts/                Optional MT-BGLR runner
scripts/
  run_training_cli.py       Main training entry point
  build_tassel_mlm_prior.py Standalone TASSEL prior utility
  export_shap_beeswarm_data.py
  convert_plink_bed_to_csv.py
  convert_vcf_to_ppmgs_csv.py
  prepare_phenotype_csv.py
examples/
  run_full_model.sh
sample_data/                Synthetic input-format examples
tests/
  test_foldwise_prior_protocol.py
docs/
  INPUT_FORMATS.md
  FOLDWISE_PRIOR.md
```

## Requirements

- Linux recommended
- Python 3.10 or 3.11
- CUDA-compatible PyTorch and an NVIDIA GPU recommended
- TASSEL 5 and Java for fold-wise MLM GWAS
- Optional: R with the `BGLR` package for the MT-BGLR baseline

## Installation

Install a PyTorch build compatible with the local CUDA driver by following the
official PyTorch installation instructions. Then install the remaining
dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, create a Conda environment with:

```bash
bash scripts/setup_ppmgs_gpu_env.sh ppmgs-gpu
conda activate ppmgs-gpu
```

Check the public command-line interface:

```bash
python scripts/run_training_cli.py --help
```

## Input data

The genotype and phenotype files are CSV matrices whose first column is
`sample_id`. Genotype marker columns must be numeric; diploid allele dosages
encoded as 0, 1, and 2 are supported. Phenotype trait columns may contain
missing cells.

```csv
sample_id,SNP_1,SNP_2,SNP_3
ind001,0,1,2
ind002,1,,0
```

```csv
sample_id,trait_1,trait_2
ind001,10.2,5.1
ind002,,4.9
```

See [docs/INPUT_FORMATS.md](docs/INPUT_FORMATS.md) for details and conversion
utilities. The files in `sample_data/` are synthetic examples only and are
intended for format and interface checks, not biological analysis.

## Preparing genotype and phenotype files

### 1. Prepare the genotype matrix

If the genotype is already a CSV matrix with one row per individual and one
column per SNP, rename the first column to `sample_id` and retain numeric marker
values. Residual missing genotypes may remain blank; PPMGS-Net imputes them
marker-wise during data preparation.

Convert PLINK BED/BIM/FAM files with:

```bash
python scripts/convert_plink_bed_to_csv.py \
  --prefix /path/to/plink_prefix \
  --output data/genotype.csv
```

Convert a VCF file with:

```bash
python scripts/convert_vcf_to_ppmgs_csv.py \
  --vcf /path/to/genotypes.vcf
```

### 2. Prepare and align the phenotype matrix

Place the individual identifier in `sample_id` and one quantitative trait in
each remaining column. Missing phenotype cells may be blank or encoded as
`NA`. Align phenotype rows to the genotype file with:

```bash
python scripts/prepare_phenotype_csv.py \
  --input /path/to/raw_phenotype.csv \
  --genotype data/genotype.csv \
  --output data/phenotype.csv
```

Sample identifiers must be unique. For multi-trait training, each retained
individual must have at least one observed target trait. Trait names supplied
through `--trait-name` or `--trait-names` must exactly match the phenotype
column names.

## Choosing a training mode

| Analysis goal | Prior settings | Transfer setting | PDAE | Missing-phenotype setting |
|---|---|---|---|---|
| Single-trait model without a prior | `--attention-mode none` | `--trait-gate-mode none` | Off | Use a complete target trait |
| Prior-aware single-trait model | `--attention-mode prior_marker_pearson` plus fold-wise prior flags | `--trait-gate-mode none` | Off | Use a complete target trait |
| Prior-only multi-trait control | Prior enabled | `--trait-gate-mode none` | Off | Add `--allow-missing` when needed |
| Prior + directional transfer | Prior enabled | `--trait-gate-mode directional_anchor` | Off | Add `--allow-missing` when needed |
| Full missing-aware PPMGS-Net | Prior enabled | `--trait-gate-mode directional_anchor` | `--pdae` | `--allow-missing` |

Use PDAE only for phenotype matrices containing missing trait cells. If
`--pdae` is requested but no phenotype cell is missing, the PDAE branch is
automatically bypassed. Directional transfer is a multi-trait component;
single-trait analyses should use `--trait-gate-mode none`.

### Choosing the prior source

For leakage-controlled cross-validation, the recommended setting is:

```text
--attention-mode prior_marker_pearson
--tassel-prior --lasso-prior
--tassel-pipeline /path/to/tassel5/run_pipeline.pl
```

This rebuilds the trait-specific TASSEL MLM and LASSO components from the
training samples of every fold. For model fitting outside cross-validation, a
compatible precomputed marker table can instead be supplied with
`--prior-marker /path/to/snp_marker.csv`. To run the no-prior control, use
`--attention-mode none` and `--trait-gate-mode none`.

The repository includes `sample_data/prior_marker.csv` as a synthetic example
of the long-format prior table. See
[docs/INPUT_FORMATS.md](docs/INPUT_FORMATS.md) for the supported long, wide,
and generic prior formats.

### Optional outputs

- Add `--save-oof-intervals` with cross-validation to export individualized
  95% out-of-fold prediction intervals.
- Add `--shap` for Top-SNP expected-gradient SHAP results.
- Add `--shap-all-markers` when the full marker attribution table is required.
- Add `--profile-resources` to record model parameter counts and peak CUDA
  memory for that run.

## Training examples

The command-line interface does not provide a default experiment seed. Choose
and pass `--seed` for each reproducible run; deterministic component-specific
offsets are applied internally.

### Quick interface check

After installing the dependencies, run a minimal no-prior analysis using the
synthetic files:

```bash
python scripts/run_training_cli.py \
  --genotype sample_data/genotype.csv \
  --phenotype sample_data/phenotype.csv \
  --out-dir results/quick_check \
  --model-family ppmgs \
  --task-type single_trait \
  --trait-name height \
  --attention-mode none \
  --trait-gate-mode none \
  --epochs 2 \
  --cv --cv-folds 2 --cv-repeats 1 \
  --seed <YOUR_SEED>
```

To check precomputed-prior parsing and directional multi-trait transfer without
installing TASSEL, run:

```bash
python scripts/run_training_cli.py \
  --genotype sample_data/genotype.csv \
  --phenotype sample_data/phenotype.csv \
  --prior-marker sample_data/prior_marker.csv \
  --out-dir results/prior_interface_check \
  --model-family ppmgs \
  --task-type multi_trait \
  --trait-names yield,height \
  --allow-missing \
  --attention-mode prior_marker_pearson \
  --attention-blend-metric pearson_learned_alpha \
  --trait-gate-mode directional_anchor \
  --prior-sparsity none \
  --epochs 2 \
  --seed <YOUR_SEED>
```

The synthetic dataset is intentionally small. It verifies file parsing and the
training interface, but it does not contain enough complete individuals to
activate the reliability-filtered PDAE teacher.

### Single-trait prediction

```bash
python scripts/run_training_cli.py \
  --genotype /path/to/genotype.csv \
  --phenotype /path/to/phenotype.csv \
  --model-family ppmgs \
  --task-type single_trait \
  --trait-name trait_1 \
  --attention-mode prior_marker_pearson \
  --attention-blend-metric pearson_learned_alpha \
  --trait-gate-mode none \
  --tassel-prior \
  --tassel-pipeline /path/to/tassel5/run_pipeline.pl \
  --lasso-prior \
  --prior-sparsity top_1pct \
  --cv --cv-folds 5 --cv-repeats 10 \
  --seed <YOUR_SEED> \
  --out-dir results/single_trait
```

### Missing-aware multi-trait prediction

```bash
python scripts/run_training_cli.py \
  --genotype /path/to/genotype.csv \
  --phenotype /path/to/phenotype_with_missing_cells.csv \
  --model-family ppmgs \
  --task-type multi_trait \
  --trait-names trait_1,trait_2 \
  --allow-missing \
  --attention-mode prior_marker_pearson \
  --attention-blend-metric pearson_learned_alpha \
  --trait-gate-mode directional_anchor \
  --pdae \
  --tassel-prior \
  --tassel-pipeline /path/to/tassel5/run_pipeline.pl \
  --lasso-prior \
  --prior-sparsity top_1pct \
  --cv --cv-folds 5 --cv-repeats 10 \
  --save-oof-intervals \
  --shap \
  --seed <YOUR_SEED> \
  --out-dir results/multi_trait
```

When cross-validation and `--tassel-prior` are enabled, TASSEL MLM and enabled
LASSO stability selection are rebuilt from the training samples of every fold.
Validation phenotypes are not used to construct, normalize, fuse, or sparsify
the fold-specific prior. See
[docs/FOLDWISE_PRIOR.md](docs/FOLDWISE_PRIOR.md).

## TPE optimization

Tune the base prior-aware model with TPE:

```bash
python scripts/run_training_cli.py \
  --genotype /path/to/genotype.csv \
  --phenotype /path/to/phenotype.csv \
  --task-type multi_trait \
  --trait-names trait_1,trait_2 \
  --allow-missing \
  --attention-mode prior_marker_pearson \
  --trait-gate-mode none \
  --tassel-prior \
  --tassel-pipeline /path/to/tassel5/run_pipeline.pl \
  --lasso-prior \
  --hyperopt --hyperopt-method tpe \
  --hyperopt-trials 50 --hyperopt-folds 2 \
  --hyperopt-metric pearson \
  --cv --cv-folds 5 --cv-repeats 10 \
  --seed <YOUR_SEED> \
  --out-dir results/tpe
```

The directional-anchor stage reuses tuned base hyperparameters. Pass the
previous result JSON through `--ppmgs-params-json` when fitting that stage.

## Outputs

Run summaries are written to `--out-dir`; model artifacts and full metadata are
stored under `saved_models/<job_id>/`. Depending on the selected options, the
workflow exports:

- Pearson correlation, RMSE, MAE, MSE, and repeated-CV summaries;
- fold-specific GWAS-LASSO prior artifacts;
- out-of-fold predictions and individualized 95% prediction intervals;
- expected-gradient SHAP marker rankings and optional all-marker tables;
- tuned hyperparameters, wall-clock timing, and optional CUDA profiling.

Generated model files and results are ignored by Git and should be archived
separately when needed.

## Protocol test

The fold-wise prior test mocks TASSEL and checks that held-out sample IDs and
phenotypes never reach the prior builder:

```bash
python tests/test_foldwise_prior_protocol.py
```

Expected output:

```text
foldwise prior protocol: PASS
```

## Data availability

Study genotype and phenotype matrices are not included in this source-code
repository. Obtain the public datasets from their original repositories and
cite the corresponding data publications. Large processed data and
supplementary result tables should be deposited in a dedicated data archive
such as Zenodo, Figshare, or OSF rather than committed to Git history.

## Citation

Publication metadata will be added after the associated manuscript receives a
stable DOI. Until then, cite the repository release or commit used for an
analysis, together with the original datasets and external software.

## License

The PPMGS-Net source code is released under the Apache License 2.0. Dataset
licenses are governed by their original providers.
