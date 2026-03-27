#!/usr/bin/env python3
"""Extract per-node initial conditions from a ParaFEM EnSight scalar file.

Accepts either:
  - A single EnSight scalar file (e.g. job.ensi.NDTTR-000010)
  - A .tar.gz archive produced by gdps_transient_thermal (picks the last NDTTR by step number)
"""
import sys
import os
import tarfile


def _extract_values(f):
    """Read float values from an open EnSight scalar file object, skipping the 4-line header."""
    values = []
    header_skipped = 0
    for line in f:
        if hasattr(line, 'decode'):
            line = line.decode('utf-8', errors='replace')
        line = line.strip()
        if header_skipped < 4:
            header_skipped += 1
            continue
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            if line.lower() in ('part', 'coordinates', 'block'):
                continue
            print(f"Warning: could not parse line: {line}")
    return values


def extract_ic(input_path, output_file):
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    values = []

    if tarfile.is_tarfile(input_path):
        with tarfile.open(input_path, 'r:gz') as tf:
            ndttr = sorted(
                [m for m in tf.getmembers() if 'NDTTR' in m.name],
                key=lambda m: m.name
            )
            if not ndttr:
                print("Error: no NDTTR files found in tarball")
                sys.exit(1)
            last = ndttr[-1]
            print(f"Using {last.name} (last NDTTR by step)")
            values = _extract_values(tf.extractfile(last))
    else:
        with open(input_path, 'r') as f:
            values = _extract_values(f)

    if not values:
        print("Error: no values found in input")
        sys.exit(1)

    with open(output_file, 'w') as f:
        f.write(f"{len(values)}\n")
        for v in values:
            f.write(f"{v:.8e}\n")

    print(f"Extracted {len(values)} IC values to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: gdps_extract_ic.py <input> <output.ini>")
        sys.exit(1)
    extract_ic(sys.argv[1], sys.argv[2])
