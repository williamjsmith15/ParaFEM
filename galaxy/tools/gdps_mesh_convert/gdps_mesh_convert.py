#!/usr/bin/env python3
import argparse
import meshio
import os
import sys

def convert_mesh(input_file, output_file, input_format=None):
    """
    Converts a mesh file to Abaqus .inp format using meshio.
    """
    try:
        print(f"Reading mesh from {input_file} (format: {input_format if input_format else 'auto'})...")
        mesh = meshio.read(input_file, file_format=input_format)
        
        # ParaFEM (via inp2pf) supports specific cell types. 
        # meshio might read types we don't want (like lines or points).
        # We filter for the primary 3D cell types supported by ParaFEM.
        supported_types = ["hexahedron", "hexahedron20", "tetra", "tetra10"]
        
        new_cells = []
        for cell_block in mesh.cells:
            if cell_block.type in supported_types:
                new_cells.append(cell_block)
            else:
                print(f"Skipping unsupported cell type: {cell_block.type}")

        if not new_cells:
            print("Error: No supported 3D cells (hex/tet) found in input mesh.")
            sys.exit(1)

        # Create a new mesh object with only supported cells
        out_mesh = meshio.Mesh(
            points=mesh.points,
            cells=new_cells,
            point_sets=mesh.point_sets,
            cell_sets=mesh.cell_sets
        )

        print(f"Writing mesh to {output_file} in abaqus format...")
        # We explicitly use 'abaqus' format for output
        out_mesh.write(output_file, file_format="abaqus")
        print("Conversion successful.")

    except Exception as e:
        print(f"Error during mesh conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert various mesh formats to Abaqus .inp for ParaFEM.")
    parser.add_argument("--input", required=True, help="Input mesh file path")
    parser.add_argument("--output", required=True, help="Output .inp file path")
    parser.add_argument("--format", help="Input mesh format (optional, meshio usually auto-detects)")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    convert_mesh(args.input, args.output, args.format)
