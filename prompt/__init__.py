import os
from utils import load_file

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "templates")
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "in_context_examples")
TRANSLATE_TEMPLATE_V1 = load_file(os.path.join(PROMPT_DIR, "translate_v1.txt"))
TRANSLATE_EXAMPLES_V1 = load_file(os.path.join(EXAMPLES_DIR, "translate_v1.txt"))
CORRECT_TEMPLATE_V1 = load_file(os.path.join(PROMPT_DIR, "correct_v1.txt"))


class PromptTemplate:
    """A prompt template."""

    def __init__(self, template: str):
        self.template: str = template

    def __call__(self, **kwargs) -> str:
        return self.template.format(**kwargs)