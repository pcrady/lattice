from typing import Callable
import math
import numpy as np
from .protein_config import ProteinConfig



EPSILON_ENERGY = 0.0


class Ensemble:
    def __init__(
        self,
        protein_string: str,
    ):
        self.protein_string: str = protein_string
        valid_chars = set("HP")
        assert set(self.protein_string) <= valid_chars

        self.n_residues: int = len(protein_string)
        assert self.n_residues >= 2

        self.t_max_topological_neighbors: int = (
            ProteinConfig.compute_t_max_topological_neighbors(self.n_residues)
        )
        self.p_min_perimeter: int = ProteinConfig.compute_p_min_perimeter(
            self.n_residues
        )

        self.ensemble: list[ProteinConfig] = self._generate_ensemble()

    def _generate_paths(self) -> list[list[tuple[int, int]]]:
        paths: list[list[tuple[int, int]]] = [[(0, 0), (1, 0)]]
        for _ in self.protein_string[2:]:
            new_paths: list[list[tuple[int, int]]] = []

            for path in paths:
                x, y = path[-1]

                candidates = [
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                ]

                for nx, ny in candidates:
                    if (nx, ny) not in path:
                        new_paths.append(path + [(nx, ny)])

            paths = new_paths

        return paths

    def _generate_ensemble(self) -> list[ProteinConfig]:
        paths = self._generate_paths()
        ensemble: list[ProteinConfig] = []
        for path in paths:
            config = []
            for i in range(len(path)):
                coordinate = path[i]
                x = coordinate[0]
                y = coordinate[1]
                value = ProteinConfig.get_residue_value(self.protein_string[i])
                cell = [x, y, value]
                config.append(cell)
            protein_config = ProteinConfig(np.array(config))
            ensemble.append(protein_config)
        return sorted(
            list(set(ensemble)), key=lambda config: config.contacts.hh_contacts
        )

    @property
    def size(self):
        return len(self.ensemble)

    @property
    def s_max_HH(self) -> int:
        least_energy_config = self.ensemble[self.size - 1]
        return least_energy_config.contacts.hh_contacts

    @property
    def degeneracies(self) -> dict[int, int]:
        degens = {}
        for m in range(self.s_max_HH + 1):
            degens[m] = self.g_degeneracy(m)
        return degens

    def G_contact_degeneracy(
        self,
        m_HH_contacts: int,
        u_non_HHcontacts: int,
    ) -> int:
        count = 0

        for ensemble in self.ensemble:
            m = ensemble.contacts.hh_contacts
            u = ensemble.contacts.hp_contacts + ensemble.contacts.pp_contacts
            if m == m_HH_contacts and u == u_non_HHcontacts:
                count = count + 1

        return count

    def g_degeneracy(
        self,
        m_HH_contacts: int,
    ) -> int:
        count = 0
        for u in range(self.t_max_topological_neighbors - m_HH_contacts + 1):
            count = count + self.G_contact_degeneracy(m_HH_contacts, u)
        return count

    def z_partition_function(
        self,
        g_degeneracy: Callable[[int], int] | None = None,
        epsilon: float = EPSILON_ENERGY,
    ) -> float:
        if g_degeneracy is None:
            g_degeneracy = self.g_degeneracy

        return sum(
            g_degeneracy(m) * math.exp((self.s_max_HH - m) * epsilon)
            for m in range(0, self.s_max_HH + 1)
        )

    def p_average_compactness(
        self,
        epsilon: float = EPSILON_ENERGY,
    ):
        value = 0
        for m in range(self.s_max_HH + 1):
            for u in range(self.t_max_topological_neighbors - m + 1):
                value = value + (
                    (u + m) / self.t_max_topological_neighbors
                ) * self.G_contact_degeneracy(m, u) * math.exp(
                    (self.s_max_HH - m) * epsilon
                )
        return value * (1 / self.z_partition_function())

    def p_native_state_avg_compactness(self):
        value = 0
        # TODO ----------------------

    def __str__(self) -> str:
        return_string: str = "---------------------------\n"
        for protein_config in self.ensemble:
            return_string = return_string + str(protein_config)
            return_string = return_string + "\n"
            return_string = return_string + "---------------------------\n"
        return return_string


proteins = [
    "HHHHHHHHHH",
    "HPHPPHPPHH",
    "HPPHPPHPHH",
    "PPPPPPHPPH",
    "PPPPPHHHHH",
    "PPPPPPPPPP",
]
for protein in proteins:
    ensemble = Ensemble(protein)
    print(ensemble)
    print(protein)
    print(ensemble.degeneracies)
    print(ensemble.z_partition_function())
    print(ensemble.p_average_compactness())
    print("")
