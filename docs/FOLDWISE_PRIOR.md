# Fold-wise GWAS-LASSO prior

Use repeated cross-validation together with both prior-construction flags:

```bash
python scripts/train_ppmgs.py \
  --genotype /path/to/genotype.csv \
  --phenotype /path/to/phenotype.csv \
  --task-type multi_trait \
  --trait-names trait_1,trait_2 \
  --allow-missing \
  --attention-mode prior_marker_pearson \
  --transfer-mode directional_anchor \
  --tassel-prior \
  --tassel-pipeline /path/to/tassel5/run_pipeline.pl \
  --lasso-prior --lasso-repeats 50 \
  --cv --cv-folds 5 --cv-repeats 10 \
  --seed 42
```

For every tuning or formal cross-validation fold, TASSEL MLM and LASSO
stability selection receive only the training samples from that fold.
Validation phenotypes are not passed to association testing, stability
selection, score normalization, prior fusion, or sparsification. A supplied
`--prior-marker` file is ignored during fold-wise evaluation when
`--tassel-prior` is active.

Fold-specific artifacts are stored under:

```text
saved_models/<job_id>/foldwise_priors/
  hyperparameter_tuning/repeat_01/fold_*/
  formal_cross_validation/repeat_*/fold_*/
```

Run the protocol test from the repository root:

```bash
python tests/test_foldwise_prior_protocol.py
```
