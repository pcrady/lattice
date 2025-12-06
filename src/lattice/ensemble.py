from typing import Callable
import math
import numpy as np
from .protein_config import ProteinConfig
import pandas as pd


# Default energy parameter for partition function calculations
# Positive values favor configurations with more HH contacts
EPSILON_ENERGY = 0.0


class Ensemble:
    """Represents an ensemble of protein configurations on a 2D lattice.
    
    The Ensemble class generates all valid self-avoiding walk configurations
    for a given protein sequence and provides methods to compute statistical
    mechanical properties such as degeneracies, partition functions, and
    average compactness.
    """
    
    def __init__(
        self,
        protein_string: str,
    ):
        """Initialize an Ensemble for a given protein sequence.
        
        Args:
            protein_string: A string of 'H' (hydrophobic) and 'P' (polar) residues.
                Must contain at least 2 residues and only 'H' and 'P' characters.
        
        Raises:
            AssertionError: If the protein string contains invalid characters
                or has fewer than 2 residues.
        """
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
        """Generate all valid self-avoiding walk paths on a 2D lattice.
        
        Generates all possible paths starting from (0,0) to (1,0) and extending
        to length n_residues, ensuring no self-intersections (self-avoiding walks).
        Each path is represented as a list of (x, y) coordinate tuples.
        
        Returns:
            A list of paths, where each path is a list of (x, y) coordinate tuples
            representing a valid self-avoiding walk configuration.
        """
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
        """Generate the complete ensemble of protein configurations.
        
        Converts all valid paths into ProteinConfig objects, assigns residue
        types based on the protein string, and removes duplicate configurations.
        The resulting ensemble is sorted by HH contact count.
        
        Returns:
            A sorted list of unique ProteinConfig objects, sorted by increasing
            HH contact count (hydrophobic-hydrophobic contacts).
        """
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
        """Get the number of unique configurations in the ensemble.
        
        Returns:
            The total number of unique protein configurations in the ensemble.
        """
        return len(self.ensemble)

    @property
    def s_max_HH(self) -> int:
        """Get the maximum number of HH contacts in the ensemble.
        
        The maximum HH contacts corresponds to the configuration with the
        lowest energy (most favorable hydrophobic interactions).
        
        Returns:
            The maximum number of hydrophobic-hydrophobic contacts found
            in any configuration in the ensemble.
        """
        least_energy_config = self.ensemble[self.size - 1]
        return least_energy_config.contacts.hh_contacts

    @property
    def degeneracies(self) -> dict[int, int]:
        """Compute the degeneracy for each possible number of HH contacts.
        
        The degeneracy g(m) represents the number of configurations with
        exactly m hydrophobic-hydrophobic contacts, summed over all possible
        non-HH contact counts.
        
        Returns:
            A dictionary mapping m (number of HH contacts) to g(m) (degeneracy).
            Keys range from 0 to s_max_HH.
        """
        degens = {}
        for m in range(self.s_max_HH + 1):
            degens[m] = self.g_degeneracy(m)
        return degens

    @property
    def degeneracies_df(self) -> pd.DataFrame:
        """Returns the degeneracies of the ensemble as a pandas dataframe.

        Returns:
            A pandas dataframe containing the degeneracies of the ensemble.
        """
        length = 6
        degens = self.degeneracies
        count = list(degens.keys())
        energy = list(degens.values())

        if len(count) < length:
            for i in range(len(count), length):
                count.append(i)
                energy.append(0)
 
        df: pd.DataFrame = pd.DataFrame({
            'count': count,
            'energy': energy,
        })

        return df

    def G_contact_degeneracy(
        self,
        m_HH_contacts: int,
        u_non_HHcontacts: int,
    ) -> int:
        """Compute the degeneracy for a specific contact configuration.
        
        Counts the number of configurations with exactly m_HH_contacts HH contacts
        and exactly u_non_HHcontacts non-HH contacts (HP + PP contacts).
        
        Args:
            m_HH_contacts: The number of hydrophobic-hydrophobic contacts.
            u_non_HHcontacts: The number of non-HH contacts (HP + PP contacts).
        
        Returns:
            The number of configurations with the specified contact counts.
        """
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
        """Equation 4

        Compute the degeneracy for a given number of HH contacts.
        
        Sums over all possible non-HH contact counts to get the total degeneracy
        for configurations with exactly m_HH_contacts HH contacts.
        
        Args:
            m_HH_contacts: The number of hydrophobic-hydrophobic contacts.
        
        Returns:
            The total number of configurations with exactly m_HH_contacts HH contacts,
            summed over all possible non-HH contact counts.
        """
        count = 0
        for u in range(self.t_max_topological_neighbors - m_HH_contacts + 1):
            count = count + self.G_contact_degeneracy(m_HH_contacts, u)
        return count

    def z_partition_function(
        self,
        g_degeneracy: Callable[[int], int] | None = None,
        epsilon: float = EPSILON_ENERGY,
    ) -> float:
        """Equation 3

        Compute the partition function Z for the ensemble.
        
        The partition function is defined as:
        Z = Σ_m g(m) * exp((s_max - m) * ε)
        
        where g(m) is the degeneracy for m HH contacts, s_max is the maximum
        HH contacts, and ε is the energy parameter.
        
        Args:
            g_degeneracy: Optional custom degeneracy function. If None, uses
                self.g_degeneracy.
            epsilon: Energy parameter ε. Defaults to EPSILON_ENERGY (0.0).
                Positive values favor configurations with more HH contacts.
        
        Returns:
            The partition function Z, a normalization constant for the ensemble.
        """
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
        """Equation 5

        Compute the average compactness of the ensemble.
        
        The average compactness is the weighted average of the compactness
        (fraction of maximum possible contacts) over all configurations,
        weighted by their Boltzmann factors.
        
        Compactness for a configuration is defined as:
        (m + u) / t_max
        
        where m is HH contacts, u is non-HH contacts, and t_max is the
        maximum possible topological neighbors.
        
        Args:
            epsilon: Energy parameter ε. Defaults to EPSILON_ENERGY (0.0).
                Positive values favor configurations with more HH contacts.
        
        Returns:
            The average compactness, a value between 0 and 1, representing
            the expected fraction of maximum possible contacts.
        """
        value = 0
        for m in range(self.s_max_HH + 1):
            for u in range(self.t_max_topological_neighbors - m + 1):
                compactness = (u + m) / self.t_max_topological_neighbors
                g = self.G_contact_degeneracy(m, u)
                exponential = math.exp((self.s_max_HH - m) * epsilon)
                value += compactness * g * exponential


 
                #value = value + (
                #    (u + m) / self.t_max_topological_neighbors
                #) * self.G_contact_degeneracy(m, u) * math.exp(
                #    (self.s_max_HH - m) * epsilon
                #)
        return value * (1 / self.z_partition_function())

    def p_native_state_avg_compactness(self):
        """Compute the average compactness of the native state.
        
        The native state is defined as the configuration(s) with the maximum
        number of HH contacts (lowest energy).
        
        Note:
            This method is currently not implemented (TODO).
        
        Returns:
            The average compactness of the native state configuration(s).
        """
        value = 0
        # TODO ----------------------

    def __str__(self) -> str:
        """Generate a string representation of the ensemble.
        
        Creates a formatted string showing all protein configurations in
        the ensemble, with separators between each configuration.
        
        Returns:
            A multi-line string representation of all configurations in
            the ensemble, separated by horizontal lines.
        """
        return_string: str = "---------------------------\n"
        for protein_config in self.ensemble:
            return_string = return_string + str(protein_config)
            return_string = return_string + "\n"
            return_string = return_string + "---------------------------\n"
        return return_string

