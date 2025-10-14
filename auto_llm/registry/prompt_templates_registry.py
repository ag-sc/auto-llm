DEFAULT_PICO_INSTRUCTION_TEMPLATE = """\
Given the text "Text", extract the PICO tags in the JSON format "Format". Do not modify the sentences.
Format:
```json
{
  "P": ["value for P"],
  "I": ["value for I"],
  "C": ["value for C"],
  "O": ["value for O"],
}
```
"""

DEFAULT_PICO_INPUT_TEMPLATE = """\
Text: {{input}}

PICO tags according to the format:
"""

DEFAULT_PICO_OUTPUT_TEMPLATE = """\
```json
{{output}}
```
"""


DEFAULT_QA_INSTRUCTION_TEMPLATE = """
Answer the following question based on the given context.
"""

DEFAULT_QA_INPUT_TEMPLATE = """\
{{input}}
"""

DEFAULT_QA_OUTPUT_TEMPLATE = """\
{{output}}
"""


INSTRUCTION_TEMPLATES_MAPPING = {
    "pico": DEFAULT_PICO_INSTRUCTION_TEMPLATE,
    "qa": DEFAULT_QA_INSTRUCTION_TEMPLATE,
}
INPUT_TEMPLATES_MAPPING = {
    "pico": DEFAULT_PICO_INPUT_TEMPLATE,
    "qa": DEFAULT_QA_INPUT_TEMPLATE,
}
OUTPUT_TEMPLATES_MAPPING = {
    "pico": DEFAULT_PICO_OUTPUT_TEMPLATE,
    "qa": DEFAULT_QA_OUTPUT_TEMPLATE,
}
