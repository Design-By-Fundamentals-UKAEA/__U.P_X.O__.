from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Define the lattice parameter for FCC copper (Cu)
a = 3.615  # Lattice parameter in Ångströms for copper
lattice = Lattice.cubic(a)

# Define the structure for copper
structure = Structure(lattice, ["Cu", "Cu", "Cu", "Cu"],
                      [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])

# Analyze the symmetry with SpacegroupAnalyzer
analyzer = SpacegroupAnalyzer(structure)

# Get and print the symmetry operations
symmetric_ops = analyzer.get_symmetry_operations()
for i, op in enumerate(symmetric_ops, start=1):
    # Corrected method to print the symmetry operation as a string
    print(f"Symmetry Operation {i}: {op}")

# Getting symmetrically equivalent positions for the first site
equivalent_sites = analyzer.get_symmetry_dataset()['equivalent_atoms']
print("Indices of symmetrically equivalent sites:", equivalent_sites)

# Get the refined structure
refined_structure = analyzer.get_refined_structure()
print("\nRefined structure:\n", refined_structure)

# Output the space group
print("Space group:", analyzer.get_space_group_symbol())
