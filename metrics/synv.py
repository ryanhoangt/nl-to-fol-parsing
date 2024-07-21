import re
from typing import List, Tuple

def is_syntactically_valid(formula: str) -> bool:
    # Remove whitespace
    formula = formula.replace(" ", "")
    
    # Check for balanced parentheses
    if not check_balanced_parentheses(formula):
        return False
    
    # Check for valid predicates
    if not check_predicates(formula):
        return False
    
    # Check for valid quantifiers
    if not check_quantifiers(formula):
        return False
    
    # Check for valid connectives
    if not check_connectives(formula):
        return False
    
    # Check for valid terms
    if not check_terms(formula):
        return False
    
    # Check for valid atomic formulas
    if not check_atomic_formulas(formula):
        return False
    
    # Check for valid complex formulas
    if not check_complex_formulas(formula):
        return False
    
    return True

def check_balanced_parentheses(formula: str) -> bool:
    stack = []
    for char in formula:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0

def check_predicates(formula: str) -> bool:
    predicates = re.findall(r'[A-Z][a-zA-Z]*\([^)]*\)', formula)
    for pred in predicates:
        if not re.match(r'^[A-Z][a-zA-Z]*\([a-z](,[a-z])*\)$', pred):
            return False
    return True

def check_quantifiers(formula: str) -> bool:
    quantifiers = re.findall(r'(∀|∃)[a-z]', formula)
    for quant in quantifiers:
        if not re.search(rf'{quant}[a-z]', formula):
            return False
    return True

def check_connectives(formula: str) -> bool:
    connectives = ['∧', '∨', '→', '↔', '¬']
    for conn in connectives:
        if conn in formula:
            # Check if connective is surrounded by valid subformulas
            parts = formula.split(conn)
            if len(parts) < 2:
                return False
            if conn != '¬' and any(not p.strip() for p in parts):
                return False
    return True

def check_terms(formula: str) -> bool:
    # Check for valid variables and constants
    terms = re.findall(r'[a-z]|[A-Z][a-zA-Z]*(?=\()', formula)
    return all(re.match(r'^[a-z]$|^[A-Z][a-zA-Z]*$', term) for term in terms)

def check_atomic_formulas(formula: str) -> bool:
    atomic_formulas = re.findall(r'[A-Z][a-zA-Z]*\([^)]*\)', formula)
    return all(re.match(r'^[A-Z][a-zA-Z]*\([a-z](,[a-z])*\)$', af) for af in atomic_formulas)

def check_complex_formulas(formula: str) -> bool:
    # Check if quantifiers are followed by a formula
    quantifier_followed_by_formula = re.findall(r'(∀|∃)[a-z]\(.*\)', formula)
    if not quantifier_followed_by_formula and ('∀' in formula or '∃' in formula):
        return False
    
    # Check if negation is followed by a formula
    negation_followed_by_formula = re.findall(r'¬\(.*\)', formula)
    if not negation_followed_by_formula and '¬' in formula:
        return False
    
    return True

def syntactic_validity(fol_formulas: List[str]) -> float:
    valid_count = sum(1 for formula in fol_formulas if is_syntactically_valid(formula))
    return valid_count / len(fol_formulas) if fol_formulas else 0


if __name__ == 'main':
    # Example usage
    fol_formulas = [
        "∀x(Human(x) → Mortal(x))",
        "Human(Socrates)",
        "Mortal(Socrates)",
        "∃y(Cat(y) ∧ Cute(y))",
        "∀x∃y(Loves(x,y))",
        "P(a) ∧ (Q(b) ∨ R(c))",
        "¬(A(x) → B(y))",
        "∀x(P(x) ↔ Q(x))",
        "Invalid(Formula",  # Invalid formula
        "∀(P(x))",  # Invalid quantifier usage
        "Predicate(x,)",  # Invalid predicate
        "A(x) + B(y)"  # Invalid connective
    ]

    synv_score = syntactic_validity(fol_formulas)
    print(f"Syntactic Validity (SynV) score: {synv_score}")