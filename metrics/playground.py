import re
from typing import List, Tuple

def is_syntactically_valid(formula: str) -> bool:
    # Remove whitespace
    formula = formula.replace(" ", "")
    
    # Check for balanced parentheses
    if formula.count('(') != formula.count(')'):
        return False
    
    # Check for valid predicates and quantifiers
    predicates = re.findall(r'[A-Z][a-zA-Z]*\([^)]*\)', formula)
    quantifiers = re.findall(r'(∀|∃)[a-z]', formula)
    
    # Check if all predicates have matching parentheses
    for pred in predicates:
        if pred.count('(') != 1 or pred.count(')') != 1:
            return False
    
    # Check if quantifiers are followed by a variable
    for quant in quantifiers:
        if not re.search(rf'{quant}[a-z]', formula):
            return False
    
    # Add more checks as needed
    
    return True

def syntactic_validity(fol_formulas: List[str]) -> float:
    valid_count = sum(1 for formula in fol_formulas if is_syntactically_valid(formula))
    return valid_count / len(fol_formulas) if fol_formulas else 0

def execute_inference_engine(premises: List[str], conclusions: List[str]) -> List[bool]:
    # Placeholder for inference engine
    # In a real scenario, you would implement or use an actual FOL inference engine
    # This is just a dummy implementation
    return [True] * len(conclusions)

def execution_accuracy(stories: List[Tuple[List[str], List[str]]]) -> float:
    total_conclusions = 0
    correct_conclusions = 0
    
    for premises, conclusions in stories:
        expected_results = [True] * len(conclusions)  # Assuming all conclusions should be true
        actual_results = execute_inference_engine(premises, conclusions)
        
        total_conclusions += len(conclusions)
        correct_conclusions += sum(1 for expected, actual in zip(expected_results, actual_results) if expected == actual)
    
    return correct_conclusions / total_conclusions if total_conclusions > 0 else 0

# Example usage
fol_formulas = [
    "∀x(Human(x) → Mortal(x))",
    "Human(Socrates)",
    "Mortal(Socrates)",
    "∃y(Cat(y) ∧ Cute(y))"
]

stories = [
    (
        ["∀x(Human(x) → Mortal(x))", "Human(Socrates)"],
        ["Mortal(Socrates)"]
    ),
    (
        ["∀x(Bird(x) → CanFly(x))", "Bird(Tweety)"],
        ["CanFly(Tweety)"]
    )
]

synv_score = syntactic_validity(fol_formulas)
eacc_score = execution_accuracy(stories)

print(f"Syntactic Validity (SynV) score: {synv_score}")
print(f"Execution Accuracy (EAcc) score: {eacc_score}")