import unittest
import numpy as np
from lattice.protein_config import ProteinConfig
from lattice.ensemble import Ensemble


class TestProteinConfig(unittest.TestCase):
    config1 = np.array(
        [
            [0, 0, 2],
            [1, 0, 1],
            [1, -1, 1],
            [1, -2, 1],
            [2, -2, 1],
            [2, -3, 2],
        ]
    )

    config2 = np.array(
        [
            [0, 0, 2],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    config3 = np.array(
        [
            [0, 0, 2],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 2],
        ]
    )

    config4 = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    config5 = np.array(
        [
            [0, 0, 2],
            [1, 0, 1],
            [1, -1, 1],
        ]
    )

    config6 = np.array(
        [
            [0, 0, 2],
            [1, 0, 1],
            [1, -1, 1],
            [1, -2, 1],
            [2, -2, 1],
            [2, -1, 2],
        ]
    )

    config7 = np.array(
        [
            [0, 0, 2],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )

    def test_residue_count(self):
        print("")
        print("-------------------------------------------------")
        prot1 = ProteinConfig(TestProteinConfig.config1)
        print(prot1)
        print(f"Prot1 - n_residues: {prot1.n_residues}")
        self.assertEqual(prot1.n_residues, 6)

    def test_t_max(self):
        print("")
        print("-------------------------------------------------")
        prot1 = ProteinConfig(TestProteinConfig.config1)
        print(
            f"Prot1 - t_max_topological_neighbors: {prot1.t_max_topological_neighbors}"
        )
        print(prot1)
        self.assertEqual(prot1.t_max_topological_neighbors, 2)

        print("")
        prot5 = ProteinConfig(TestProteinConfig.config5)
        print(
            f"Prot5 - t_max_topological_neighbors: {prot5.t_max_topological_neighbors}"
        )
        print(prot5)
        self.assertEqual(prot5.t_max_topological_neighbors, 0)

    def test_topological_contacts(self):
        print("")
        print("-------------------------------------------------")
        prot1 = ProteinConfig(TestProteinConfig.config1)
        print(f"Prot1 Topological Contacts:")
        print(prot1)
        print(prot1.contacts)
        self.assertEqual(prot1.contacts.hh_contacts, 0)
        self.assertEqual(prot1.contacts.hp_contacts, 0)
        self.assertEqual(prot1.contacts.pp_contacts, 0)

        print("")
        prot2 = ProteinConfig(TestProteinConfig.config2)
        print(f"Prot2 Topological Contacts:")
        print(prot2)
        print(prot2.contacts)
        self.assertEqual(prot2.contacts.hh_contacts, 0)
        self.assertEqual(prot2.contacts.hp_contacts, 1)
        self.assertEqual(prot2.contacts.pp_contacts, 0)

        print("")
        prot3 = ProteinConfig(TestProteinConfig.config3)
        print(f"Prot3 Topological Contacts:")
        print(prot3)
        print(prot3.contacts)
        self.assertEqual(prot3.contacts.hh_contacts, 1)
        self.assertEqual(prot3.contacts.hp_contacts, 0)
        self.assertEqual(prot3.contacts.pp_contacts, 0)

        print("")
        prot4 = ProteinConfig(TestProteinConfig.config4)
        print(f"Prot4 Topological Contacts:")
        print(prot4)
        print(prot4.contacts)
        self.assertEqual(prot4.contacts.hh_contacts, 0)
        self.assertEqual(prot4.contacts.hp_contacts, 0)
        self.assertEqual(prot4.contacts.pp_contacts, 1)

    def test_compactness(self):
        print("")
        print("-------------------------------------------------")
        prot1 = ProteinConfig(TestProteinConfig.config1)
        print(prot1)
        print(f"Prot1 - compactness: {prot1.compactness}")
        self.assertEqual(prot1.compactness, 0)


        print("")
        prot2 = ProteinConfig(TestProteinConfig.config2)
        print(prot2)
        print(f"Prot2 - compactness: {prot2.compactness}")
        self.assertEqual(prot2.compactness, 1)

        print("")
        prot6 = ProteinConfig(TestProteinConfig.config6)
        print(prot6)
        print(f"Prot2 - compactness: {prot6.compactness}")
        self.assertEqual(prot6.compactness, 0.5)

    def test_equality(self):
        print("")
        print("-------------------------------------------------")
        prot1 = ProteinConfig(TestProteinConfig.config5)
        prot2 = ProteinConfig(TestProteinConfig.config7)

        print(prot1)
        print(prot2)
        self.assertEqual(prot1, prot2)

    def test_ensemble_number(self):
        print("")
        print("-------------------------------------------------")
        ensemble = Ensemble("HPPPPPPPPP")
        print(f"Ensemble size for n = 10: {ensemble.size}")
        self.assertEqual(ensemble.size, 2034)

    def test_degeneracies(self):
        print("")
        print("-------------------------------------------------")
        proteins = ['HHHHHHHHHH', 'HPHPPHPPHH', 'HPPHPPHPHH', 'PPPPPPHPPH', 'PPPPPHHHHH', 'PPPPPPPPPP']
        zeros = []
        expected_zeros = [666, 1435, 1164, 1688, 1461, 2034]
        for protein in proteins:
            ensemble = Ensemble(protein)
            zeros.append(ensemble.degeneracies[0])

        for i in range(len(zeros)):
            self.assertEqual(zeros[i], expected_zeros[i])



if __name__ == "__main__":
    unittest.main()
