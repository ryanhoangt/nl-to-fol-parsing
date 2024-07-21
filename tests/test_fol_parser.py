import unittest
from fol_parser import parse_text_FOL_to_tree

class TestParseTextFOLToTree(unittest.TestCase):

    def test_valid_formula(self):
        formula = "A(x) ∧ B(x)"
        root_node = parse_text_FOL_to_tree(formula)
        self.assertIsNotNone(root_node)

if __name__ == '__main__':
    unittest.main()

