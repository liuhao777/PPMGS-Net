# Input formats

## Genotype matrix

The genotype file is a comma-separated matrix. The first column must be
`sample_id`; each remaining column is a marker. Marker values must be numeric.
Diploid allele dosages encoded as 0, 1, and 2 are supported. Blank cells and
common missing-value tokens are imputed marker-wise by the training pipeline.

```csv
sample_id,SNP_1,SNP_2,SNP_3
ind001,0,1,2
ind002,1,,0
```

## Phenotype matrix

The phenotype file is a comma-separated matrix. The first column must be
`sample_id`; each remaining selected column is a quantitative trait. Missing
phenotype cells may be blank or use `NA`.

```csv
sample_id,trait_1,trait_2
ind001,10.2,5.1
ind002,,4.9
```

Genotype and phenotype rows are matched by `sample_id`. Duplicate identifiers
are not allowed.

## Optional prior table

A precomputed marker-prior table must use marker identifiers that match the
genotype columns. For leakage-controlled cross-validation, prefer the
fold-wise `--tassel-prior --lasso-prior` workflow instead of a prior built from
the full phenotype matrix.

## Conversion utilities

- `scripts/convert_plink_bed_to_csv.py` converts PLINK BED/BIM/FAM files.
- `scripts/convert_vcf_to_ppmgs_csv.py` converts VCF genotype calls.
- `scripts/prepare_phenotype_csv.py` aligns a phenotype table to genotype IDs.

The files under `sample_data/` are synthetic format examples and are not study
data.
