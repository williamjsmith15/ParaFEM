"""
Stitch multiple transient VTU archives from IC chain runs into one time series.

Each input is a .tar.gz from gdps_parafem2vtu_transient containing a .pvd
orchestration file and per-timestep .vtu files. The timestamps from each
segment are offset by the cumulative end time of all preceding segments so
the output timeline is continuous.

The final state of each segment becomes the IC for the next — the VTU outputs
themselves do not overlap, so no deduplication is needed.
"""

import argparse
import os
import re
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from xml.dom import minidom


def parse_pvd(pvd_path):
    tree = ET.parse(pvd_path)
    entries = []
    for ds in tree.iter('DataSet'):
        entries.append((float(ds.attrib['timestep']), ds.attrib['file']))
    return sorted(entries)


def write_pvd(entries, pvd_path):
    root = ET.Element('VTKFile', type='Collection', version='1.0', byte_order='LittleEndian')
    coll = ET.SubElement(root, 'Collection')
    for t, f in entries:
        ET.SubElement(coll, 'DataSet', timestep=f'{t:.6g}', part='0', file=f)
    raw = ET.tostring(root, encoding='unicode')
    pretty = minidom.parseString(raw).toprettyxml(indent='  ')
    lines = [l for l in pretty.splitlines() if not l.startswith('<?xml')]
    with open(pvd_path, 'w') as fh:
        fh.write('\n'.join(lines))


def collect_segment(archive, seg_dir):
    os.makedirs(seg_dir)
    with tarfile.open(archive, 'r:*') as tar:
        tar.extractall(seg_dir)

    pvd_files = [f for f in os.listdir(seg_dir) if f.endswith('.pvd')]
    if pvd_files:
        return parse_pvd(os.path.join(seg_dir, pvd_files[0]))

    # Single-timestep archive with no PVD — treat as t=0
    vtu_files = sorted(f for f in os.listdir(seg_dir) if f.endswith('.vtu'))
    if not vtu_files:
        raise RuntimeError(f'No .pvd or .vtu files found in {archive}')
    return [(0.0, vtu_files[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='append', required=True,
                        help='VTU tar.gz archive, one per IC segment in order')
    parser.add_argument('--output', required=True, help='Output tar.gz path')
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp()
    try:
        all_frames = []   # (global_time, src_path, global_step_number)
        time_offset = 0.0
        global_step = 1

        for seg_idx, archive in enumerate(args.input):
            seg_dir = os.path.join(tmpdir, f'seg_{seg_idx:03d}')
            entries = collect_segment(archive, seg_dir)

            for local_t, local_fname in entries:
                src = os.path.join(seg_dir, local_fname)
                all_frames.append((local_t + time_offset, src, global_step))
                global_step += 1

            time_offset += max(t for t, _ in entries)

        out_dir = os.path.join(tmpdir, 'output')
        os.makedirs(out_dir)
        pvd_entries = []

        for global_t, src, step_num in all_frames:
            dest_name = f'stitched_{step_num:06d}.vtu'
            shutil.copy2(src, os.path.join(out_dir, dest_name))
            pvd_entries.append((global_t, dest_name))

        write_pvd(pvd_entries, os.path.join(out_dir, 'stitched.pvd'))

        with tarfile.open(args.output, 'w:gz') as tar:
            for fname in sorted(os.listdir(out_dir)):
                tar.add(os.path.join(out_dir, fname), arcname=fname)

        print(f'Stitched {len(pvd_entries)} timesteps from {len(args.input)} segments.')
        print(f'Time range: {pvd_entries[0][0]:.4g} — {pvd_entries[-1][0]:.4g} s')

    finally:
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
