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


INSTRUCTION_TEMPLATES_MAPPING = {"pico": DEFAULT_PICO_INSTRUCTION_TEMPLATE}
INPUT_TEMPLATES_MAPPING = {"pico": DEFAULT_PICO_INPUT_TEMPLATE}
OUTPUT_TEMPLATES_MAPPING = {"pico": DEFAULT_PICO_OUTPUT_TEMPLATE}
