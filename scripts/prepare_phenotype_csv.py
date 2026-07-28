from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NA_VALUES = ["", "NA", "NaN", "nan", "null", "None"]


def prepare(input_path: Path, genotype_path: Path, output_path: Path) -> None:
    phenotype = pd.read_csv(input_path, na_values=NA_VALUES, keep_default_na=True)
    if phenotype.empty:
        raise ValueError(f"{input_path} is empty.")

    first_col = phenotype.columns[0]
    if "sample_id" not in phenotype.columns:
        phenotype = phenotype.rename(columns={first_col: "sample_id"})

    if "sample_id" not in phenotype.columns:
        raise ValueError("Phenotype CSV must contain a sample_id column or an ID-like first column.")

    phenotype["sample_id"] = phenotype["sample_id"].astype(str).str.strip()
    phenotype = phenotype[phenotype["sample_id"].notna() & (phenotype["sample_id"] != "")]
    if phenotype["sample_id"].duplicated().any():
        duplicated = phenotype.loc[phenotype["sample_id"].duplicated(), "sample_id"].head(10).tolist()
        raise ValueError(f"Duplicated phenotype sample_id values found: {duplicated}")

    trait_cols = [col for col in phenotype.columns if col != "sample_id"]
    if not trait_cols:
        raise ValueError("No phenotype trait columns were found.")

    for col in trait_cols:
        phenotype[col] = pd.to_numeric(phenotype[col], errors="coerce")

    genotype_ids = pd.read_csv(genotype_path, usecols=["sample_id"])["sample_id"].astype(str).str.strip()
    genotype_order = pd.DataFrame({"sample_id": genotype_ids})

    phenotype_ids = set(phenotype["sample_id"])
    genotype_id_set = set(genotype_order["sample_id"])
    missing_in_phenotype = [sid for sid in genotype_order["sample_id"] if sid not in phenotype_ids]
    extra_in_phenotype = [sid for sid in phenotype["sample_id"] if sid not in genotype_id_set]

    aligned = genotype_order.merge(phenotype, on="sample_id", how="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)

    observed_counts = aligned[trait_cols].notna().sum().to_dict()
    report_path = output_path.with_suffix(".prepare_report.txt")
    report_path.write_text(
        "\n".join(
            [
                f"input_phenotype={input_path}",
                f"genotype_reference={genotype_path}",
                f"output={output_path}",
                f"genotype_samples={len(genotype_order)}",
                f"phenotype_samples={len(phenotype)}",
                f"output_samples={len(aligned)}",
                f"traits={','.join(trait_cols)}",
                f"missing_in_phenotype_count={len(missing_in_phenotype)}",
                f"extra_in_phenotype_count={len(extra_in_phenotype)}",
                f"missing_in_phenotype_first10={','.join(missing_in_phenotype[:10])}",
                f"extra_in_phenotype_first10={','.join(extra_in_phenotype[:10])}",
                "observed_trait_values=" + ",".join(f"{trait}:{observed_counts[trait]}" for trait in trait_cols),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align a phenotype CSV to a PPMGS-Net genotype CSV.")
    parser.add_argument("--input", required=True, help="Raw phenotype CSV path.")
    parser.add_argument("--genotype", required=True, help="Converted genotype CSV path used for sample_id ordering.")
    parser.add_argument("--output", required=True, help="Output phenotype CSV path.")
    args = parser.parse_args()
    prepare(Path(args.input), Path(args.genotype), Path(args.output))


if __name__ == "__main__":
    main()
