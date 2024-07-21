import unittest
from metrics import is_syntactically_valid

class TestIsSyntacticallyValid(unittest.TestCase):

    def test_valid_formula(self):
        formula = "A & B"
        self.assertTrue(is_syntactically_valid(formula))

    def test_invalid_formula(self):
        formula = "A &"
        self.assertFalse(is_syntactically_valid(formula))

    def test_empty_string(self):
        formula = ""
        self.assertFalse(is_syntactically_valid(formula))

    def test_special_characters(self):
        formula = "A & B @"
        self.assertFalse(is_syntactically_valid(formula))

    def test_nested_formula(self):
        formula = "(A & B) | C"
        self.assertTrue(is_syntactically_valid(formula))

    def test_formula_with_quantifiers(self):
        formula = "∀x (P(x) → Q(x))"
        self.assertTrue(is_syntactically_valid(formula))

    def test_formula_with_negation(self):
        formula = "¬(A ∨ B)"
        self.assertTrue(is_syntactically_valid(formula))

    def test_formula_with_multiple_operators(self):
        formula = "A ∧ B ∨ C → D"
        self.assertTrue(is_syntactically_valid(formula))

    def test_fol_formulas(self):
        valid_formulas = [
            "∀x(Human(x) → Mortal(x))",
            "Human(Socrates)",
            "Mortal(Socrates)",
            "∃y(Cat(y) ∧ Cute(y))",
            "∀x∃y(Loves(x,y))",
            "P(a) ∧ (Q(b) ∨ R(c))",
            "¬(A(x) → B(y))",
            "∀x(P(x) ↔ Q(x))"
        ]
        invalid_formulas = [
            "Invalid(Formula",
            "∀(P(x))",
            "Predicate(x,)",
            "A(x) + B(y)"
        ]
        for formula in valid_formulas:
            with self.subTest(formula=formula):
                self.assertTrue(is_syntactically_valid(formula))
        for formula in invalid_formulas:
            with self.subTest(formula=formula):
                self.assertFalse(is_syntactically_valid(formula))

if __name__ == '__main__':
    unittest.main()

