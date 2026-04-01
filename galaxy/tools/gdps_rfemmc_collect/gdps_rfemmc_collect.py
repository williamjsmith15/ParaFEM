#!/usr/bin/env python3
import argparse
import json
import os
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re

# Add parent dir for parafem_common
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from parafem_common import parse_d_file
except ImportError:
    # Fallback if parafem_common is not in PYTHONPATH or adjacent
    def parse_d_file(filepath):
        nodes = {}
        section = None
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line == '*NODES': section = 'nodes'; continue
                if line == '*ELEMENTS': section = 'elements'; continue
                if section == 'nodes':
                    parts = line.split()
                    if len(parts) >= 4:
                        nodes[int(parts[0])] = [float(parts[1]), float(parts[2]), float(parts[3])]
        return nodes, None, None, None

def extract_flux(ensi_tar_path, nodes, z_min, dz):
    """
    Extracts the flux proxy from the EnSight scalar field.
    J = sum(values[nid] for nid in layer1) / len(layer1) / dz
    where layer1 is the first layer of nodes above z_min.
    """
    with tarfile.open(ensi_tar_path) as tar:
        ndptl_member = None
        for m in tar.getmembers():
            if 'NDPTL' in m.name:
                ndptl_member = m
                break
        if not ndptl_member:
            return None
        with tar.extractfile(ndptl_member) as f:
            lines = f.read().decode().splitlines()
            
    # Values start after 4-line header
    values = {}
    for i, line in enumerate(lines[4:]):
        node_id = i + 1
        values[node_id] = float(line.strip())
        
    # Layer 1 nodes: nodes at z_min + dz
    layer1_nodes = [nid for nid, xyz in nodes.items() if abs(xyz[2] - (z_min + dz)) < 1e-12]
    
    if not layer1_nodes:
        # Fallback to downstream nodes if layer1 is empty (should not happen for a mesh)
        # but user's snippet used downstream (z_min).
        # We use layer1 because downstream values are typically 0.0 (BC).
        layer1_nodes = [nid for nid, xyz in nodes.items() if abs(xyz[2] - z_min) < 1e-12]

    if not layer1_nodes:
        return 0.0
        
    flux = sum(values[nid] for nid in layer1_nodes) / len(layer1_nodes) / dz
    return flux

def main():
    parser = argparse.ArgumentParser(description="Aggregate RFEM MC results")
    parser.add_argument('--d_file', required=True)
    parser.add_argument('--res_file', action='append', nargs=2, metavar=('ID', 'PATH'), help='Instance ID and .res path')
    parser.add_argument('--ensi_file', action='append', nargs=2, metavar=('ID', 'PATH'), help='Instance ID and ensi.tar.gz path')
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--output_png', required=True)
    
    args = parser.parse_args()
    
    # Parse mesh to get node coords and dz
    nodes, _, _, _ = parse_d_file(args.d_file)
    coords = np.array(list(nodes.values()))
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    z_coords = sorted(list(set(coords[:, 2])))
    if len(z_coords) > 1:
        dz = z_coords[1] - z_coords[0]
    else:
        dz = z_max - z_min if z_max > z_min else 1.0

    # Build maps of ID -> path
    res_map = {id: path for id, path in args.res_file} if args.res_file else {}
    ensi_map = {id: path for id, path in args.ensi_file} if args.ensi_file else {}
    
    # Instance IDs are keys present in both or either
    instance_ids = sorted(list(set(res_map.keys()) | set(ensi_map.keys())))
    
    results = []
    for inst_id in instance_ids:
        res_path = res_map.get(inst_id)
        ensi_path = ensi_map.get(inst_id)
        
        qoi = None
        if ensi_path and os.path.exists(ensi_path):
            try:
                qoi = extract_flux(ensi_path, nodes, z_min, dz)
            except Exception as e:
                print(f"Error extracting flux for instance {inst_id}: {e}")
        
        # Optionally parse .res for iteration count or convergence
        iters = None
        if res_path and os.path.exists(res_path):
            with open(res_path, 'r') as f:
                content = f.read()
                match = re.search(r'iterations to convergence was\s+(\d+)', content)
                if match:
                    iters = int(match.group(1))
        
        results.append({
            'instance_id': inst_id,
            'flux': qoi,
            'iterations': iters
        })
        
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    
    # Statistics
    stats = {
        'count': int(df['flux'].count()) if 'flux' in df else 0,
        'mean': float(df['flux'].mean()) if 'flux' in df else 0.0,
        'std': float(df['flux'].std()) if 'flux' in df else 0.0,
        'min': float(df['flux'].min()) if 'flux' in df else 0.0,
        'max': float(df['flux'].max()) if 'flux' in df else 0.0,
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(stats, f, indent=4)
        
    # Plotting
    if not df['flux'].dropna().empty:
        plt.figure(figsize=(10, 6))
        plt.hist(df['flux'].dropna(), bins=20, color='skyblue', edgecolor='black')
        plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=1, label=f"Mean: {stats['mean']:.3e}")
        plt.title('Distribution of Downstream Flux Across MC Instances')
        plt.xlabel('Flux proxy (J)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(args.output_png, format='png')
    else:
        # Create empty plot if no data
        plt.figure()
        plt.text(0.5, 0.5, 'No flux data available', ha='center', va='center')
        plt.savefig(args.output_png, format='png')

if __name__ == "__main__":
    main()
