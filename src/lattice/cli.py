#!/usr/bin/env python3
import argparse
import sys
import threading
import time
from pathlib import Path

# Add src directory to path if running without installation
# This allows the script to work both as a module and as a direct script
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parent.parent  # src/ directory
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from lattice.ensemble import Ensemble


class LoadingIndicator:
    def __init__(self):
        self.spinner_chars = ['/', '-', '\\', '|']
        self.stop_flag = threading.Event()
        self.thread = None
    
    def _spin(self):
        i = 0
        while not self.stop_flag.is_set():
            sys.stdout.write(f'\rProcessing... {self.spinner_chars[i % len(self.spinner_chars)]}')
            sys.stdout.flush()
            time.sleep(0.2)
            i += 1
        sys.stdout.write('\r' + ' ' * 20 + '\r')  # Clear the line
        sys.stdout.flush()
    
    def start(self):
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join()


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
        
        loader = None
        try:
            # Start loading indicator
            loader = LoadingIndicator()
            loader.start()
            
            ensemble = Ensemble(sequence)
            
            # Stop loading indicator
            loader.stop()
            loader = None
            
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
            if loader:
                loader.stop()
            print(f"Error processing sequence '{sequence}': {e}")
            continue


if __name__ == "__main__":
    main()

