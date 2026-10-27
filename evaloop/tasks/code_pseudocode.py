"""Code -> structured pseudocode; no original requirements are shown to the model."""
from .code_summarization import CodeSummarizationTask
from .factory import register_task

PSEUDOCODE_PROMPT = '''Translate the following Python code into language-independent pseudocode.
Describe its actual behavior using explicit INPUT, OUTPUT, and numbered STEPS.
Preserve conditions, loop bounds, ordering, duplicate handling, return types,
and exceptional or empty-input behavior present in the code. Do not repair the
algorithm, add requirements, or reproduce Python source. Return only pseudocode.

```python
{code}
```'''


@register_task("code_pseudocode")
class CodePseudocodeTask(CodeSummarizationTask):
    def create_prompt(self, input_text):
        return PSEUDOCODE_PROMPT.format(code=input_text)

    def postprocess(self, output, context):
        # Always recover the anchor from the immutable original prompt.
        original = context.get("initial_prompt", context.get("cycle_input", ""))
        assertions = [line.strip() for line in original.splitlines()
                      if line.strip().startswith("assert ")]
        if not assertions:
            raise ValueError("Original prompt must contain an assertion anchor")
        return '\"\"\"\n' + output.strip() + '\n' + assertions[-1] + '\n\"\"\"\n'
