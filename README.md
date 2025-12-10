# Lattice Protein Folding Simulation

A Python package for simulating protein folding on 2D lattices, computing ensemble properties, and analyzing protein conformations.

## Features

- Generate all valid self-avoiding walk configurations for protein sequences
- Compute statistical mechanical properties (partition functions, degeneracies, compactness)
- Analyze native states and their properties
- Visualize protein configurations
- Interactive protein folding with the `fold()` method

## Installation

### Using Conda (Recommended)

1. Clone or download this repository:
   ```bash
   cd lattice
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate lattice
   ```

3. The package is automatically installed in editable mode when the environment is created.

### Using pip

1. Clone or download this repository:
   ```bash
   cd lattice
   ```

2. Install the package and dependencies:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from lattice.ensemble import Ensemble
from lattice.protein_config import ProteinConfig

# Create an ensemble for a protein sequence
ensemble = Ensemble("HPPHPPHPHH")

# Access properties
print(f"Maximum HH contacts: {ensemble.s_max_HH}")
print(f"Average compactness: {ensemble.p_average_compactness_native_state}")

# Create a protein configuration from a string
protein = ProteinConfig.from_string("HPPHPPHPHH")

# Fold the protein
protein.fold()
print(protein)  # Visualize the folded structure
```

### Running the Jupyter Notebook

1. Activate the conda environment:
   ```bash
   conda activate lattice
   ```

2. Start Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `paper_replication.ipynb` and run the cells.

### Interactive Folding

```python
from lattice.protein_config import ProteinConfig
import time

# Create a protein
protein = ProteinConfig.from_string("HHHHHHHHHHPPPPPPPPPP")

# Fold it multiple times
for i in range(10):
    print(f"\nFold {i+1}:")
    print(protein)
    print(protein.contacts)
    protein.fold()
    time.sleep(0.5)
```

## Project Structure

```
lattice/
├── src/
│   └── lattice/
│       ├── __init__.py
│       ├── protein_config.py    # Protein configuration and folding
│       ├── ensemble.py           # Ensemble generation and analysis
│       └── cli.py                # Command-line interface
├── tests/
│   └── test_lattice.py
├── paper_replication.ipynb       # Main analysis notebook
├── environment.yml               # Conda environment file
├── requirements.txt              # pip requirements
├── pyproject.toml                # Package configuration
└── README.md                     # This file
```

## Dependencies

- Python >= 3.10
- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0
- ipython >= 7.0.0

## Development

To contribute or modify the code:

1. Install in editable mode (already done with conda environment):
   ```bash
   pip install -e .
   ```

2. Run tests:
   ```bash
   pytest
   ```

## License

See LICENSE file for details.

## Author

Peter Crady

