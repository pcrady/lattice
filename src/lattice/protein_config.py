import math
import numpy as np
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


