import multiprocessing
from fol_parser import parse_text_FOL_to_tree

def is_syntactically_valid(formula: str) -> bool:
    if not formula.strip():
        return False
    root_node = parse_text_FOL_to_tree(formula)

    return root_node is not None

def is_execution_accurate() -> bool:
    # TODO:
    pass

def fol_syntax_metric_will_timeout(formula: str, timeout: int) -> bool:
    p = multiprocessing.Process(target=is_syntactically_valid, args=(formula,))
    p.start()
    p.join(timeout=timeout)
    p.terminate()
    if p.exitcode is None:
        return True
    return False

def is_syntactically_valid_with_timeout(formula: str) -> bool:
    if fol_syntax_metric_will_timeout(formula, 10):
        return True
    return is_syntactically_valid(formula)