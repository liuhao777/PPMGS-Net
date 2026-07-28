from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def split_sample_id(sample: str, mode: str) -> str:
    sample = sample.strip()
    if len(sample) >= 2 and sample[0] == sample[-1] and sample[0] in {"'", '"'}:
        sample = sample[1:-1].strip()
    if mode == "full":
        return sample
    if mode == "prefix":
        return sample.split("-", 1)[0].strip() or sample
    if mode == "sample_prefix":
        if "_L" in sample:
            return sample.split("_L", 1)[0].strip() or sample
        return sample.split("_", 1)[0].strip() or sample
    raise ValueError(f"Unknown sample ID mode: {mode}")


def make_unique(values: list[str]) -> tuple[list[str], int]:
    seen: Counter[str] = Counter()
    unique = []
    renamed = 0
    for value in values:
        base = str(value).strip() or "item"
        seen[base] += 1
        if seen[base] == 1:
            unique.append(base)
        else:
            renamed += 1
            unique.append(f"{base}_{seen[base]}")
    return unique, renamed


def marker_name(chrom: str, pos: str, source_id: str, ref: str, alt: str, mode: str) -> str:
    source_id = source_id.strip()
    if mode == "id" and source_id and source_id != ".":
        return source_id
    clean_alt = alt.split(",", 1)[0].strip()
    clean_ref = ref.strip()
    return f"{chrom.strip()}_{pos.strip()}_{clean_ref}_{clean_alt}"


def gt_to_dosage(value: str) -> float:
    token = (value or "").strip()
    if not token or token in {".", "./.", ".|.", "NA", "NaN", "nan"}:
        return np.nan
    gt = token.split(":", 1)[0]
    if "/" in gt:
        parts = gt.split("/")
    elif "|" in gt:
        parts = gt.split("|")
    else:
        parts = [gt]

    dosage = 0
    for part in parts:
        if part in {"", "."}:
            return np.nan
        if not part.isdigit():
            return np.nan
        dosage += 0 if int(part) == 0 else 1
    return float(dosage)


def read_vcf_layout(path: Path, marker_mode: str, sample_id_mode: str) -> tuple[list[str], list[str], list[dict[str, str]], list[dict[str, str]]]:
    samples: list[str] | None = None
    raw_samples: list[str] | None = None
    raw_marker_names: list[str] = []
    marker_rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n\r").split("\t")
                raw_samples = header[9:]
                samples = [split_sample_id(sample, sample_id_mode) for sample in raw_samples]
                continue
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 10:
                continue
            chrom, pos, source_id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            name = marker_name(chrom, pos, source_id, ref, alt, marker_mode)
            raw_marker_names.append(name)
            marker_rows.append(
                {
                    "marker": name,
                    "chrom": chrom,
                    "pos": pos,
                    "source_id": source_id,
                    "ref": ref,
                    "alt": alt,
                }
            )

    if samples is None or raw_samples is None:
        raise ValueError(f"No #CHROM header was found in {path}.")
    marker_names, renamed_markers = make_unique(raw_marker_names)
    sample_names, renamed_samples = make_unique(samples)
    if renamed_markers:
        for row, marker in zip(marker_rows, marker_names):
            row["marker"] = marker
    sample_rows = [
        {"sample_id": sample_id, "vcf_sample_id": raw_sample}
        for sample_id, raw_sample in zip(sample_names, raw_samples)
    ]
    return sample_names, marker_names, marker_rows, sample_rows


def convert_vcf(path: Path, output: Path, marker_map: Path, sample_map: Path, sample_id_mode: str, marker_mode: str) -> dict[str, object]:
    sample_names, marker_names, marker_rows, sample_rows = read_vcf_layout(path, marker_mode, sample_id_mode)
    values = np.empty((len(sample_names), len(marker_names)), dtype=np.float32)
    values[:] = np.nan

    marker_index = 0
    raw_tokens: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n\r").split("\t")
            sample_values = parts[9:]
            if len(sample_values) != len(sample_names):
                raise ValueError(
                    f"Sample count mismatch at marker {marker_index + 1}: "
                    f"expected {len(sample_names)}, found {len(sample_values)}"
                )
            for sample_index, token in enumerate(sample_values):
                values[sample_index, marker_index] = gt_to_dosage(token)
                if marker_index < 500:
                    raw_tokens[token.split(":", 1)[0]] += 1
            marker_index += 1

    if marker_index != len(marker_names):
        raise ValueError(f"Expected {len(marker_names)} markers but parsed {marker_index}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    marker_map.parent.mkdir(parents=True, exist_ok=True)
    sample_map.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *marker_names])
        for sample_id, row in zip(sample_names, values):
            writer.writerow([sample_id, *("" if np.isnan(value) else int(value) for value in row)])

    pd.DataFrame(marker_rows).to_csv(marker_map, index=False)
    pd.DataFrame(sample_rows).to_csv(sample_map, index=False)

    observed = int(np.isfinite(values).sum())
    total = int(values.size)
    return {
        "source": str(path),
        "output": str(output),
        "marker_map": str(marker_map),
        "sample_map": str(sample_map),
        "samples": len(sample_names),
        "markers": len(marker_names),
        "observed_genotypes": observed,
        "missing_genotypes": total - observed,
        "missing_rate": (total - observed) / max(total, 1),
        "sample_id_mode": sample_id_mode,
        "marker_name_mode": marker_mode,
        "raw_gt_tokens": dict(raw_tokens.most_common(12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert real VCF files to PPMGS-Net genotype CSV.")
    parser.add_argument("--vcf", nargs="+", required=True, type=Path, help="One or more .vcf files.")
    parser.add_argument("--sample-id-mode", choices=["prefix", "full", "sample_prefix"], default="prefix")
    parser.add_argument("--marker-mode", choices=["chrom_pos_ref_alt", "id"], default="chrom_pos_ref_alt")
    parser.add_argument("--suffix", default="_genotype.csv", help="Output genotype CSV suffix.")
    args = parser.parse_args()

    summaries = []
    for vcf_path in args.vcf:
        if not vcf_path.exists():
            raise FileNotFoundError(vcf_path)
        output = vcf_path.with_name(f"{vcf_path.stem}{args.suffix}")
        marker_map = vcf_path.with_name(f"{vcf_path.stem}_marker_map.csv")
        sample_map = vcf_path.with_name(f"{vcf_path.stem}_sample_map.csv")
        summary = convert_vcf(
            vcf_path,
            output,
            marker_map,
            sample_map,
            sample_id_mode=args.sample_id_mode,
            marker_mode=args.marker_mode,
        )
        summaries.append(summary)
        report = vcf_path.with_name(f"{vcf_path.stem}_conversion_report.txt")
        report.write_text("\n".join(f"{key}={value}" for key, value in summary.items()) + "\n", encoding="utf-8")
        print(f"Converted: {output}")

    print("\nSummary")
    for item in summaries:
        print(
            f"{Path(str(item['source'])).name}: "
            f"samples={item['samples']}, markers={item['markers']}, "
            f"missing_rate={float(item['missing_rate']):.6f}"
        )


if __name__ == "__main__":
    main()
