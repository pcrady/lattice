import math
import numpy as np
from scipy.sparse import coo_matrix
from collections import OrderedDict


class Contacts:
    def __init__(self):
        self.hh_contacts: int = 0
        self.hp_contacts: int = 0
        self.pp_contacts: int = 0

    def __str__(self) -> str:
        return (
            f"HH: {self.hh_contacts}\n"
            f"HP: {self.hp_contacts}\n"
            f"PP: {self.pp_contacts}"
        )

    def increment(self, value):
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
    def __init__(
        self,
        config: np.ndarray,
        t_max_topological_neighbors: int | None = None,
    ):
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
        return n_residues + 1 - (ProteinConfig.compute_p_min_perimeter(n_residues) // 2)

    @staticmethod
    def compute_p_min_perimeter(n_residues) -> int:
        m = math.ceil(math.sqrt(n_residues) - 1)

        if m**2 < n_residues and n_residues <= m * (m + 1):
            return 4 * m + 2
        elif m * (m + 1) < n_residues and n_residues <= (m + 1) ** 2:
            return 4 * m + 4
        else:
            raise Exception("Error in Pmin calculation")

    @staticmethod
    def get_residue_value(char: str) -> int:
        if char == "H":
            return 2
        elif char == "P":
            return 1
        else:
            raise Exception("Error invalid character in protein_string")

    def _compute_contacts(self) -> None:
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
        if self.t_max_topological_neighbors == 0:
            return 1
        return (
            self.contacts.hh_contacts
            + self.contacts.hp_contacts
            + self.contacts.pp_contacts
        ) / self.t_max_topological_neighbors

    def _label(self, value: int) -> str:
        if value == 0:
            return "."
        elif value == 1:
            return "P"
        elif value == 2:
            return "H"
        else:
            raise Exception(f"Error illegal lattice value: {value}")

    def __str__(self) -> str:
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
        direct = self.config.tobytes()
        reflected = self.config.copy()
        reflected[:, 1] = -reflected[:, 1]
        refl_bytes = reflected.tobytes()

        key = min(direct, refl_bytes)
        return hash((self.config.shape, key))

    def __eq__(self, other):
        if not isinstance(other, ProteinConfig):
            return NotImplemented
        x_reflected_config = self.config.copy()
        x_reflected_config[:, 1] = -x_reflected_config[:, 1]
        return np.array_equal(self.config, other.config) or np.array_equal(
            x_reflected_config, other.config
        )


