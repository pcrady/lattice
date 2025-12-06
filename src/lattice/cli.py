#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src directory to path if running without installation
# This allows the script to work both as a module and as a direct script
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parent.parent  # src/ directory
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from lattice.ensemble import Ensemble


def main():
    parser = argparse.ArgumentParser(
        description="Calculate degeneracies, partition function, and average compactness for protein sequences"
    )
    parser.add_argument(
        "--proteins", "-p",
        nargs="+",
        required=True,
        help="One or more protein sequences (e.g., HHHPPHP)",
    )
    
    args = parser.parse_args()
    
    for sequence in args.proteins:
        print(f"\n{'='*60}")
        print(f"Sequence: {sequence}")
        print(f"{'='*60}")
        
        try:
            ensemble = Ensemble(sequence)
            
            # Output degeneracies
            print("\nDegeneracies:")
            degeneracies = ensemble.degeneracies
            for m, count in sorted(degeneracies.items()):
                print(f"  m={m}: {count}")
            
            # Output partition function
            z = ensemble.z_partition_function()
            print(f"\nZ (Partition Function): {z}")
            
            # Output average compactness
            p_avg = ensemble.p_average_compactness()
            print(f"P (Average Compactness): {p_avg}")
            
        except Exception as e:
            print(f"Error processing sequence '{sequence}': {e}")
            continue


if __name__ == "__main__":
    main()

