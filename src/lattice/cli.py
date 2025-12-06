#!/usr/bin/env python3
import argparse
import sys
import threading
import time
import traceback
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
        self.spinner_chars = ["/", "-", "\\", "|"]
        self.stop_flag = threading.Event()
        self.thread = None

    def _spin(self):
        i = 0
        while not self.stop_flag.is_set():
            sys.stdout.write(
                f"\rProcessing... {self.spinner_chars[i % len(self.spinner_chars)]}"
            )
            sys.stdout.flush()
            time.sleep(0.2)
            i += 1
        sys.stdout.write("\r" + " " * 20 + "\r")  # Clear the line
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
        "--proteins",
        "-p",
        nargs="+",
        required=True,
        help="One or more protein sequences (e.g., HHHPPHP)",
    )
    parser.add_argument(
        "--ensemble",
        "-e",
        action="store_true",
        help="Print the ensemble configurations for each protein",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Write output to a file instead of console (loading indicator still shows on console)",
    )

    args = parser.parse_args()

    # Open file if specified, otherwise use stdout
    output_file = None
    if args.file:
        output_file = open(args.file, "w")

    # Helper function to print to file or stdout
    def output_print(*args, **kwargs):
        if output_file:
            print(*args, file=output_file, **kwargs)
            output_file.flush()
        else:
            print(*args, **kwargs)

    try:
        for sequence in args.proteins:
            output_print(f"\n{'='*60}")
            output_print(f"Sequence: {sequence}")
            output_print(f"{'='*60}")

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
                output_print("\nDegeneracies:")
                degeneracies = ensemble.degeneracies
                for m, count in sorted(degeneracies.items()):
                    output_print(f"  m={m}: {count}")

                # Output partition function
                z = ensemble.z_partition_function()
                output_print(f"\nZ (Partition Function): {z}")

                # Output average compactness
                p_avg = ensemble.p_average_compactness()
                output_print(f"P (Average Compactness): {p_avg}")

                # Output ensemble if flag is set
                if args.ensemble:
                    output_print("\nEnsemble Configurations:")
                    output_print(str(ensemble))

            except Exception as e:
                if loader:
                    loader.stop()
                output_print(f"Error processing sequence '{sequence}': {e}")
                output_print("\nTraceback:")
                # Print traceback to the same output destination
                if output_file:
                    traceback.print_exc(file=output_file)
                    output_file.flush()
                else:
                    traceback.print_exc()
                continue
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
