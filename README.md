# PPMGS-Net

PPMGS-Net is a command-line implementation of prior-informed and
phenotype-missing-aware multi-trait genomic prediction.

The released workflow contains the method reported in the accompanying
manuscript:

- trait-specific GWAS-LASSO marker priors;
- a genome-wide path and a trait-specific prior-gated path with learned fusion;
- independently trained and frozen trait anchors;
- bounded, low-rank asymmetric directional residual transfer;
- cross-fitted genomic-PDAE teacher predictions with reliability and
  confidence filtering for missing phenotype cells;
- repeated cross-validation, TPE optimization, out-of-fold individualized
  prediction intervals, and expected-gradient SHAP output.

Study datasets, generated priors, model checkpoints, training logs,
manuscript files, historical experiments, and server-specific launch scripts
are not included in this source-code repository.

## Repository layout

```text
backend/app/
  training.py               Final PPMGS-Net model and training workflow
  gwas_prior.py             TASSEL MLM prior construction
scripts/
  train_ppmgs.py            Main command-line entry point
  build_tassel_mlm_prior.py Standalone TASSEL prior utility
  export_shap_beeswarm_data.py
  convert_plink_bed_to_csv.py
  convert_vcf_to_ppmgs_csv.py
  prepare_phenotype_csv.py
  setup_ppmgs_gpu_env.sh
examples/
  run_full_model.sh
sample_data/                Synthetic input-format examples
tests/
  test_cli_smoke.py
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

Check the command-line interface:

```bash
python scripts/train_ppmgs.py --help
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

See [docs/INPUT_FORMATS.md](docs/INPUT_FORMATS.md) for details. Files under
`sample_data/` are synthetic examples for format and interface checks only.

### Preparing input matrices

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

Align a phenotype table to genotype sample order with:

```bash
python scripts/prepare_phenotype_csv.py \
  --input /path/to/raw_phenotype.csv \
  --genotype data/genotype.csv \
  --output data/phenotype.csv
```

Sample identifiers must be unique. Each retained multi-trait individual must
have at least one observed target trait. Names passed through `--trait-name`
or `--trait-names` must exactly match phenotype columns.

## Training modes

| Analysis | Prior | Transfer | PDAE |
|---|---|---|---|
| Genotype-only single trait | `--attention-mode none` | `--transfer-mode none` | Off |
| Prior-aware single trait | `--attention-mode prior_marker_pearson` | `--transfer-mode none` | Off |
| Prior-only multi-trait control | Prior enabled | `--transfer-mode none` | Off |
| Prior plus directional transfer | Prior enabled | `--transfer-mode directional_anchor` | Off |
| Full missing-aware PPMGS-Net | Prior enabled | `--transfer-mode directional_anchor` | `--pdae` |

Directional transfer requires at least two traits. PDAE is used only when the
training phenotype contains missing target cells; it is bypassed automatically
when no target phenotype is missing.

## Prior construction

For fold-wise prior construction during cross-validation, use:

```text
--attention-mode prior_marker_pearson
--tassel-prior --lasso-prior
--tassel-pipeline /path/to/tassel5/run_pipeline.pl
```

TASSEL MLM and LASSO stability components are rebuilt from the training samples
of each tuning or formal cross-validation fold. A compatible precomputed marker
table can instead be supplied through `--prior-marker` for fitting outside the
fold-wise workflow. See [docs/FOLDWISE_PRIOR.md](docs/FOLDWISE_PRIOR.md).

## Examples

Every reproducible run requires an explicit `--seed`.

### Quick interface check

```bash
python scripts/train_ppmgs.py \
  --genotype sample_data/genotype.csv \
  --phenotype sample_data/phenotype.csv \
  --out-dir results/quick_check \
  --task-type single_trait \
  --trait-name height \
  --attention-mode none \
  --transfer-mode none \
  --epochs 2 \
  --cv --cv-folds 2 --cv-repeats 1 \
  --seed 42
```

### Prior-aware single-trait prediction

```bash
python scripts/train_ppmgs.py \
  --genotype /path/to/genotype.csv \
  --phenotype /path/to/phenotype.csv \
  --task-type single_trait \
  --trait-name trait_1 \
  --attention-mode prior_marker_pearson \
  --transfer-mode none \
  --tassel-prior \
  --tassel-pipeline /path/to/tassel5/run_pipeline.pl \
  --lasso-prior --lasso-repeats 50 \
  --prior-sparsity top_1pct \
  --cv --cv-folds 5 --cv-repeats 10 \
  --seed 42 \
  --out-dir results/single_trait
```

### Missing-aware multi-trait prediction

```bash
python scripts/train_ppmgs.py \
  --genotype /path/to/genotype.csv \
  --phenotype /path/to/phenotype_with_missing_cells.csv \
  --task-type multi_trait \
  --trait-names trait_1,trait_2 \
  --allow-missing \
  --attention-mode prior_marker_pearson \
  --transfer-mode directional_anchor \
  --pdae \
  --tassel-prior \
  --tassel-pipeline /path/to/tassel5/run_pipeline.pl \
  --lasso-prior --lasso-repeats 50 \
  --prior-sparsity top_1pct \
  --cv --cv-folds 5 --cv-repeats 10 \
  --save-oof-intervals \
  --shap \
  --seed 42 \
  --out-dir results/multi_trait
```

### TPE optimization

TPE is the only hyperparameter-search method exposed by the final workflow:

```bash
python scripts/train_ppmgs.py \
  --genotype /path/to/genotype.csv \
  --phenotype /path/to/phenotype.csv \
  --task-type multi_trait \
  --trait-names trait_1,trait_2 \
  --allow-missing \
  --attention-mode prior_marker_pearson \
  --transfer-mode none \
  --tassel-prior \
  --tassel-pipeline /path/to/tassel5/run_pipeline.pl \
  --lasso-prior \
  --hyperopt --hyperopt-trials 50 --hyperopt-folds 2 \
  --hyperopt-metric pearson \
  --cv --cv-folds 5 --cv-repeats 10 \
  --seed 42 \
  --out-dir results/tpe
```

The directional-transfer stage can reuse tuned anchor hyperparameters through
`--ppmgs-params-json`.

## Optional outputs

- `--save-oof-intervals`: individualized 95% out-of-fold prediction intervals.
- `--shap`: Top-SNP expected-gradient SHAP results.
- `--shap-all-markers`: complete marker attribution table.
- `--profile-resources`: parameter count and peak CUDA memory.

Run summaries are written to `--out-dir`; model artifacts and full metadata are
stored under `saved_models/<job_id>/`.

## Tests

```bash
python tests/test_cli_smoke.py
python tests/test_foldwise_prior_protocol.py
```

The fold-wise prior protocol test verifies that held-out sample identifiers and
phenotypes do not reach the prior builder.

## Data availability

Study genotype and phenotype matrices are not included. Obtain public datasets
from their original repositories and cite the corresponding data publications.
Large processed data and result tables should be deposited in a dedicated data
archive rather than committed to Git history.

## Citation

Publication metadata will be added after the associated manuscript receives a
stable DOI. Until then, cite the repository release or commit used for an
analysis, together with the original datasets and external software.

## License

PPMGS-Net is released under the Apache License 2.0. Dataset licenses are
governed by their original providers.
