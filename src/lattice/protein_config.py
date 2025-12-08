import math
import numpy as np
import random
from scipy.sparse import coo_matrix
from collections import OrderedDict


class Contacts:
    """Tracks contact counts between different residue types.

    A contact is defined as a non-adjacent neighbor interaction on the lattice.
    This class counts the number of contacts between:
    - HH: hydrophobic-hydrophobic contacts (value = 4)
    - HP: hydrophobic-polar contacts (value = 2)
    - PP: polar-polar contacts (value = 1)
    """

    def __init__(self):
        """Initialize a Contacts object with zero contacts."""
        self.hh_contacts: int = 0
        self.hp_contacts: int = 0
        self.pp_contacts: int = 0

    def __str__(self) -> str:
        """Generate a string representation of contact counts.

        Returns:
            A multi-line string showing HH, HP, and PP contact counts.
        """
        return (
            f"HH: {self.hh_contacts}\n"
            f"HP: {self.hp_contacts}\n"
            f"PP: {self.pp_contacts}"
        )

    def increment(self, value):
        """Increment the appropriate contact counter based on residue product.

        The value parameter represents the product of two residue values:
        - 4 = H * H (hydrophobic-hydrophobic contact)
        - 2 = H * P or P * H (hydrophobic-polar contact)
        - 1 = P * P (polar-polar contact)
        - 0 = empty space (no contact)

        Args:
            value: The product of two residue values (0, 1, 2, or 4).

        Raises:
            Exception: If value is not 0, 1, 2, or 4.
        """
        if value == 4:
            self.hh_contacts += 1
        elif value == 2:
            self.hp_contacts += 1
        elif value == 1:
            self.pp_contacts += 1
        elif value == 0:
            pass
        else:
            raise Exception(f"Error illegal lattice value: {value}")


class ProteinConfig:
    """Represents a single protein configuration on a 2D lattice.

    A ProteinConfig stores the coordinates and residue types for a protein
    folded on a 2D square lattice. It computes contact interactions and
    provides methods for visualization and comparison.

    The configuration is stored as a numpy array with shape (n_residues, 3),
    where each row contains [x, y, residue_value].
    """

    def __init__(
        self,
        config: np.ndarray,
        t_max_topological_neighbors: int | None = None,
    ):
        """Initialize a ProteinConfig from a configuration array.

        Args:
            config: A numpy array of shape (n_residues, 3) where each row
                contains [x, y, residue_value]. Residue values: 2 for H,
                1 for P, 0 for empty.
            t_max_topological_neighbors: Maximum number of topological neighbors.
                If None, computed automatically from n_residues.

        Note:
            The configuration is automatically shifted to start at (0,0) and
            contacts are computed during initialization.
        """
        self.config: np.ndarray = config
        self.shifted_config: np.ndarray = self.config.copy()
        self.n_residues: int = len(config)

        if t_max_topological_neighbors == None:
            self.t_max_topological_neighbors: int = (
                ProteinConfig.compute_t_max_topological_neighbors(self.n_residues)
            )
        else:
            self.t_max_topological_neighbors = t_max_topological_neighbors

        self.contacts: Contacts = Contacts()
        self._initialized_for_folding: bool = False  # Track if initialized for folding
        self._compute_contacts()

    def generate_lattice(self) -> np.ndarray:
        """Generate a 2D lattice representation of the protein configuration.

        Creates a 2D array where each cell contains the residue value at that
        position. The lattice is shifted to start at (0,0) and flipped vertically
        for display (y-axis inverted).

        Returns:
            A 2D numpy array representing the lattice, with values:
            - 2 for hydrophobic (H) residues
            - 1 for polar (P) residues
            - 0 for empty cells
        """
        x_coords = self.config[:, 0]
        y_coords = self.config[:, 1]
        vals = self.config[:, 2]

        x_min, y_min = x_coords.min(), y_coords.min()
        x_coords = x_coords - x_min
        y_coords = y_coords - y_min

        self.shifted_config[:, 0] -= x_min
        self.shifted_config[:, 1] -= y_min

        height = y_coords.max() + 1
        width = x_coords.max() + 1

        lattice = coo_matrix(
            (vals, (y_coords, x_coords)), shape=(height, width)
        ).toarray()
        return np.flipud(lattice)

    @staticmethod
    def compute_t_max_topological_neighbors(n_residues: int) -> int:
        """Equation 1

        Compute the maximum number of topological neighbors for a protein.

        The maximum topological neighbors is calculated as:
        t_max = n_residues + 1 - (p_min / 2)

        where p_min is the minimum perimeter for n_residues.

        Args:
            n_residues: The number of residues in the protein.

        Returns:
            The maximum number of topological neighbors possible for a protein
            with n_residues residues.
        """
        return n_residues + 1 - (ProteinConfig.compute_p_min_perimeter(n_residues) // 2)

    @staticmethod
    def compute_p_min_perimeter(n_residues) -> int:
        """Equation 2

        Compute the minimum perimeter for a compact shape with n_residues.

        The minimum perimeter is calculated based on the optimal rectangular
        or near-rectangular arrangement of residues. This is used to determine
        the maximum possible contacts.

        Args:
            n_residues: The number of residues in the protein.

        Returns:
            The minimum perimeter for a compact shape containing n_residues.

        Raises:
            Exception: If the calculation fails (should not occur for valid inputs).
        """
        m = math.ceil(math.sqrt(n_residues) - 1)

        if m**2 < n_residues and n_residues <= m * (m + 1):
            return 4 * m + 2
        elif m * (m + 1) < n_residues and n_residues <= (m + 1) ** 2:
            return 4 * m + 4
        else:
            raise Exception("Error in Pmin calculation")

    @staticmethod
    def get_residue_value(char: str) -> int:
        """Convert a residue character to its numerical value.

        Args:
            char: A single character representing the residue type.
                'H' for hydrophobic, 'P' for polar.

        Returns:
            The numerical value: 2 for 'H', 1 for 'P'.

        Raises:
            Exception: If char is not 'H' or 'P'.
        """
        if char == "H":
            return 2
        elif char == "P":
            return 1
        else:
            raise Exception("Error invalid character in protein_string")

    @staticmethod
    def from_string(protein_string: str) -> "ProteinConfig":
        """Generate a protein configuration along the x-axis from a string.

        Creates a ProteinConfig initialized along the x-axis starting at (0,0).
        The protein is laid out as: (0,0), (1,0), (2,0), ..., (n-1, 0).

        Args:
            protein_string: A string of 'H' (hydrophobic) and 'P' (polar) residues.
                Must contain only 'H' and 'P' characters and have at least 2 residues.

        Returns:
            A ProteinConfig instance with the protein initialized along the x-axis.

        Raises:
            ValueError: If protein_string is empty, has fewer than 2 residues,
                or contains invalid characters.
        """
        if not protein_string:
            raise ValueError("Protein string cannot be empty")
        
        if len(protein_string) < 2:
            raise ValueError("Protein string must have at least 2 residues")
        
        valid_chars = set("HP")
        if not set(protein_string) <= valid_chars:
            raise ValueError(f"Protein string can only contain 'H' and 'P' characters")
        
        # Create configuration array along x-axis
        n_residues = len(protein_string)
        config = np.zeros((n_residues, 3), dtype=int)
        
        for i, char in enumerate(protein_string):
            config[i] = [i, 0, ProteinConfig.get_residue_value(char)]
        
        protein = ProteinConfig(config)
        protein._initialized_for_folding = True  # Mark as initialized for folding
        return protein

    def _compute_contacts(self) -> None:
        """Compute all contacts between non-adjacent residues.

        A contact is defined as two residues that are neighbors on the lattice
        (horizontally or vertically adjacent) but are not consecutive in the
        protein sequence (sequence distance > 1).

        Contacts are checked in two directions:
        - Right neighbor: (x+1, y)
        - Bottom neighbor: (x, y-1)

        The contact type is determined by the product of residue values:
        - 4 = H*H (hydrophobic-hydrophobic)
        - 2 = H*P or P*H (hydrophobic-polar)
        - 1 = P*P (polar-polar)
        """
        coords = OrderedDict(((x, y), v) for x, y, v in self.shifted_config)
        coord_index: dict[tuple[int, int], int] = {
            (x, y): i for i, (x, y, _) in enumerate(self.shifted_config)
        }

        x_max = self.shifted_config[:, 0].max()
        y_min = self.shifted_config[:, 1].min()

        for i, (x, y, value) in enumerate(self.shifted_config):
            if x < x_max:
                coord_n = (x + 1, y)
                neighbor = coords.get(coord_n)
                if neighbor is not None:
                    j = coord_index[coord_n]
                    if abs(i - j) > 1:
                        self.contacts.increment(value * neighbor)

            if y > y_min:
                coord_n = (x, y - 1)
                neighbor = coords.get(coord_n)
                if neighbor is not None:
                    j = coord_index[coord_n]
                    if abs(i - j) > 1:
                        self.contacts.increment(value * neighbor)

    @property
    def compactness(self) -> float:
        """Compute the compactness of the protein configuration.

        Compactness is defined as the fraction of maximum possible contacts:
        compactness = (total_contacts) / t_max_topological_neighbors

        where total_contacts = HH + HP + PP contacts.

        Returns:
            A value between 0 and 1, where 1 represents maximum compactness
            (all possible contacts formed). Returns 1 if t_max is 0.
        """
        if self.t_max_topological_neighbors == 0:
            return 1
        return (
            self.contacts.hh_contacts
            + self.contacts.hp_contacts
            + self.contacts.pp_contacts
        ) / self.t_max_topological_neighbors

    def interior_h_fraction(self) -> float:
        """Compute the fraction of interior residues that are hydrophobic (H).

        A residue is considered interior if all 4 neighbors (up, down, left, right)
        are occupied by other residues. This measures the degree to which H residues
        are partitioned into a solvophobic core.

        The fraction x is defined as:
        x = n_hi / n_i

        where:
        - n_hi is the number of hydrophobic (H) residues in the interior
        - n_i is the total number of interior residues

        Returns:
            The fraction x, a value between 0 and 1. Returns 0.0 if there are
            no interior residues (n_i = 0).
        """
        occupied_coords = set((x, y) for x, y, _ in self.shifted_config)

        n_hi = 0
        n_i = 0

        for x, y, value in self.shifted_config:
            neighbors = [
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ]

            if all(neighbor in occupied_coords for neighbor in neighbors):
                n_i += 1
                if value == 2:
                    n_hi += 1

        if n_i == 0:
            return 0.0

        return n_hi / n_i

    @property
    def turn_vector(self) -> np.ndarray:
        """Compute the turn vector representation of the chain conformation.

        Each bond pair (three consecutive residues) is represented by:
        - 0: collinear (straight)
        - +1: right turn (clockwise)
        - -1: left turn (counterclockwise)

        The conformation is represented by the vector:
        v = [b1, b2, ..., b_{n-2}]

        where n is the number of residues. This has the useful feature that
        the mirror image conformation is represented simply by -v.

        Returns:
            A numpy array of shape (n_residues - 2,) containing the turn
            values for each bond pair. Each value is -1, 0, or +1.
        """
        if self.n_residues < 3:
            return np.array([], dtype=int)

        turn_values = []
        for i in range(self.n_residues - 2):
            p1 = self.shifted_config[i]
            p2 = self.shifted_config[i + 1]
            p3 = self.shifted_config[i + 2]

            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            x3, y3 = p3[0], p3[1]

            dx1 = x2 - x1
            dy1 = y2 - y1
            dx2 = x3 - x2
            dy2 = y3 - y2

            cross = dx1 * dy2 - dy1 * dx2

            if cross == 0:
                turn_values.append(0)
            elif cross > 0:
                turn_values.append(-1)
            else:
                turn_values.append(1)

        return np.array(turn_values, dtype=int)

    def distance(self, other: "ProteinConfig") -> int:
        """Equation 13

        Compute the distance between two conformations using their turn vectors.

        The distance measure d(v1, v2) between two conformations is:
        d(v1, v2) = min[c(v1 - v2), c(v1 + v2)]

        where c represents the operation of summing the absolute values of
        the elements. The minimum accounts for the fact that the mirror image
        of a conformation is represented by -v, so we consider both the
        difference and the sum (which corresponds to comparing with the
        mirror image).

        Args:
            other: Another ProteinConfig to compute the distance to.

        Returns:
            The distance between the two conformations, a non-negative integer.
            Returns 0 if the conformations are identical or mirror images.
        """
        v1 = self.turn_vector
        v2 = other.turn_vector

        if len(v1) != len(v2):
            raise ValueError(
                f"Cannot compute distance: conformations have different lengths "
                f"({len(v1)} vs {len(v2)} turn values)"
            )

        diff = np.abs(v1 - v2).sum()
        sum_abs = np.abs(v1 + v2).sum()

        return int(min(diff, sum_abs))

    def _is_initialized_along_x_axis(self) -> bool:
        """Check if the protein is initialized along the x-axis starting at (0,0).
        
        Returns:
            True if all residues have y=0 and x coordinates increase from 0.
        """
        if len(self.config) == 0:
            return False
        
        # Check if all y coordinates are 0
        if not np.all(self.config[:, 1] == 0):
            return False
        
        # Check if x coordinates are 0, 1, 2, 3, ...
        x_coords = self.config[:, 0]
        expected_x = np.arange(len(self.config))
        return np.array_equal(x_coords, expected_x)

    def _has_self_intersection(self, config: np.ndarray) -> bool:
        """Check if a configuration has self-intersection (overlapping coordinates).
        
        Args:
            config: A numpy array of shape (n_residues, 3) with [x, y, value] rows.
            
        Returns:
            True if any coordinate appears more than once (self-intersection).
        """
        coords = [(row[0], row[1]) for row in config]
        return len(coords) != len(set(coords))

    def fold(self, max_attempts: int = 100) -> bool:
        """Fold the protein by bending at a random location.
        
        If the protein has not been initialized for folding, initializes it along
        the x-axis first. Then randomly picks a location along the protein and bends
        it at a right angle in a random direction. Retries if the bend causes
        self-intersection.
        
        The bend is performed by rotating the chain segment after the chosen location
        by 90 degrees in a random direction (clockwise or counterclockwise).
        Subsequent calls to fold() will continue folding from the current state.
        
        Args:
            max_attempts: Maximum number of attempts to find a valid fold that doesn't
                cause self-intersection. Defaults to 100.
                
        Returns:
            True if a valid fold was successfully applied, False if no valid fold
            could be found after max_attempts attempts.
        """
        # Initialize along x-axis only once (first time fold is called)
        if not self._initialized_for_folding:
            # Initialize: (0,0), (1,0), (2,0), ..., (n-1, 0)
            new_config = np.zeros((self.n_residues, 3), dtype=int)
            for i in range(self.n_residues):
                new_config[i] = [i, 0, self.config[i, 2]]  # Preserve residue values
            self.config = new_config
            self.shifted_config = self.config.copy()
            self._initialized_for_folding = True
            self._compute_contacts()
        
        # Need at least 3 residues to fold (need a segment to rotate)
        if self.n_residues < 3:
            return False
        
        # Pick a random location to bend (not at the ends)
        # We'll bend after position bend_index, so bend_index can be 0 to n-2
        bend_index = random.randint(0, self.n_residues - 2)
        
        for attempt in range(max_attempts):
            # Create a copy to test the fold
            test_config = self.config.copy()
            
            # Determine the current direction from bend_index to bend_index+1
            dx = test_config[bend_index + 1, 0] - test_config[bend_index, 0]
            dy = test_config[bend_index + 1, 1] - test_config[bend_index, 1]
            
            # Choose rotation direction (clockwise or counterclockwise)
            clockwise = random.random() < 0.5
            
            # Apply the bend: rotate the segment starting at bend_index+1
            # The pivot point is at bend_index
            pivot_x = test_config[bend_index, 0]
            pivot_y = test_config[bend_index, 1]
            
            # Update all positions after the bend_index
            for i in range(bend_index + 1, self.n_residues):
                # Calculate relative position from the pivot
                rel_x = test_config[i, 0] - pivot_x
                rel_y = test_config[i, 1] - pivot_y
                
                # Rotate the relative position by 90 degrees
                if clockwise:
                    # Clockwise: (x, y) -> (y, -x)
                    new_rel_x = rel_y
                    new_rel_y = -rel_x
                else:
                    # Counterclockwise: (x, y) -> (-y, x)
                    new_rel_x = -rel_y
                    new_rel_y = rel_x
                
                # Update the position
                test_config[i, 0] = pivot_x + new_rel_x
                test_config[i, 1] = pivot_y + new_rel_y
            
            # Check if this causes self-intersection
            if not self._has_self_intersection(test_config):
                # Also check that consecutive residues are still neighbors
                valid = True
                for i in range(self.n_residues - 1):
                    dx_check = abs(test_config[i + 1, 0] - test_config[i, 0])
                    dy_check = abs(test_config[i + 1, 1] - test_config[i, 1])
                    if dx_check + dy_check != 1:  # Not adjacent
                        valid = False
                        break
                
                if valid:
                    # Valid fold! Update the configuration
                    self.config = test_config
                    self.shifted_config = self.config.copy()
                    self._compute_contacts()
                    return True
        
        # Could not find a valid fold after max_attempts
        return False

    def _label(self, value: int) -> str:
        """Convert a numerical residue value to its character label.

        Args:
            value: The numerical residue value (0, 1, or 2).

        Returns:
            A single character: 'H' for 2, 'P' for 1, '.' for 0.

        Raises:
            Exception: If value is not 0, 1, or 2.
        """
        if value == 0:
            return "."
        elif value == 1:
            return "P"
        elif value == 2:
            return "H"
        else:
            raise Exception(f"Error illegal lattice value: {value}")

    def __str__(self) -> str:
        """Generate a visual string representation of the protein configuration.

        Creates an ASCII art representation showing:
        - Residue positions (H, P, or .)
        - Bonds between consecutive residues (| for vertical, - for horizontal)

        The lattice is displayed with y-axis inverted (top to bottom).

        Returns:
            A multi-line string showing the 2D lattice configuration with
            bonds between residues.
        """
        lattice = self.generate_lattice()
        height, width = lattice.shape

        out_h = 2 * height - 1
        out_w = 2 * width - 1
        grid: list[list[str]] = [[" " for _ in range(out_w)] for _ in range(out_h)]

        for r in range(height):
            for c in range(width):
                grid[2 * r][2 * c] = self._label(lattice[r, c])

        for (x1, y1, _), (x2, y2, _) in zip(
            self.shifted_config, self.shifted_config[1:]
        ):
            if abs(x1 - x2) + abs(y1 - y2) != 1:
                continue

            r1 = height - 1 - y1
            c1 = x1
            r2 = height - 1 - y2
            c2 = x2

            ar1, ac1 = 2 * r1, 2 * c1
            ar2, ac2 = 2 * r2, 2 * c2
            cr = (ar1 + ar2) // 2
            cc = (ac1 + ac2) // 2

            if x1 != x2:
                grid[cr][cc] = "-"
            else:
                grid[cr][cc] = "|"

        return "\n".join(" ".join(row) for row in grid)

    def __hash__(self):
        """Compute a hash value for the protein configuration.

        The hash accounts for rotational/reflectional symmetry by using
        the minimum of the direct configuration and its y-axis reflection.
        This ensures that configurations that are mirror images of each other
        have the same hash.

        Returns:
            An integer hash value for the configuration.
        """
        direct = self.config.tobytes()
        reflected = self.config.copy()
        reflected[:, 1] = -reflected[:, 1]
        refl_bytes = reflected.tobytes()

        key = min(direct, refl_bytes)
        return hash((self.config.shape, key))

    def __eq__(self, other):
        """Check if two protein configurations are equivalent.

        Two configurations are considered equal if they are identical or
        if one is the y-axis reflection of the other. This accounts for
        the symmetry of the lattice model.

        Args:
            other: Another ProteinConfig object to compare.

        Returns:
            True if the configurations are equivalent, False otherwise.
            Returns NotImplemented if other is not a ProteinConfig.
        """
        if not isinstance(other, ProteinConfig):
            return NotImplemented
        x_reflected_config = self.config.copy()
        x_reflected_config[:, 1] = -x_reflected_config[:, 1]
        return np.array_equal(self.config, other.config) or np.array_equal(
            x_reflected_config, other.config
        )
