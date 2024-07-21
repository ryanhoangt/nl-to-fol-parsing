from fol_parser import parse_text_FOL_to_tree

def is_syntactically_valid(formula: str) -> bool:
    if not formula.strip():
        return False
    root_node = parse_text_FOL_to_tree(formula)

    return root_node is not None

def is_execution_accurate() -> bool:
    # TODO:
    pass
