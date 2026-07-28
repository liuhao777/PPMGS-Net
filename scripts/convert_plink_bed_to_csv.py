from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bed_reader import open_bed


def read_fam_ids(fam_path: Path) -> list[str]:
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, dtype=str)
    if fam.shape[1] < 2:
        raise ValueError(f"{fam_path} does not look like a PLINK .fam file.")
    fid = fam.iloc[:, 0].astype(str)
    iid = fam.iloc[:, 1].astype(str)
    sample_ids = iid.where(fid.isin(["0", iid]), fid + "_" + iid).tolist()
    if len(set(sample_ids)) != len(sample_ids):
        sample_ids = (fid + "_" + iid).tolist()
    return sample_ids


def read_bim_markers(bim_path: Path) -> list[str]:
    bim = pd.read_csv(bim_path, sep=r"\s+", header=None, dtype=str)
    if bim.shape[1] < 2:
        raise ValueError(f"{bim_path} does not look like a PLINK .bim file.")
    marker_names = bim.iloc[:, 1].astype(str).tolist()
    if len(set(marker_names)) == len(marker_names):
        return marker_names

    seen: dict[str, int] = {}
    unique_names = []
    for name in marker_names:
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return unique_names


def convert(prefix: Path, output: Path) -> None:
    fam_path = prefix.with_suffix(".fam")
    bim_path = prefix.with_suffix(".bim")
    bed_path = prefix.with_suffix(".bed")
    for path in (fam_path, bim_path, bed_path):
        if not path.exists():
            raise FileNotFoundError(path)

    sample_ids = read_fam_ids(fam_path)
    marker_names = read_bim_markers(bim_path)
    bed = open_bed(bed_path)
    values = bed.read(dtype="float32")

    if values.shape != (len(sample_ids), len(marker_names)):
        raise ValueError(
            "BED shape does not match FAM/BIM files: "
            f"bed={values.shape}, fam={len(sample_ids)}, bim={len(marker_names)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    genotype = pd.DataFrame(values, columns=marker_names)
    genotype.insert(0, "sample_id", sample_ids)
    genotype.to_csv(output, index=False)

    report = output.with_suffix(".conversion_report.txt")
    report.write_text(
        "\n".join(
            [
                f"source_prefix={prefix}",
                f"fam={fam_path}",
                f"bim={bim_path}",
                f"bed={bed_path}",
                f"output={output}",
                f"samples={len(sample_ids)}",
                f"markers={len(marker_names)}",
                "encoding=0/1/2 allele dosage from PLINK BED via bed-reader",
                "missing_genotypes=blank cells in CSV",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PLINK bed/bim/fam to a PPMGS-Net genotype CSV.")
    parser.add_argument("--prefix", required=True, help="PLINK file prefix without the .bed/.bim/.fam suffixes.")
    parser.add_argument("--output", required=True, help="Output genotype CSV path")
    args = parser.parse_args()
    convert(Path(args.prefix), Path(args.output))


if __name__ == "__main__":
    main()
