# Evaluating

Different metrics can be used to evaluate performances of LLMs, depending on the specific task.

## Sequence to Label

### Accuracy

Accuracy measures the proportion of inputs the model classified correctly, expressed as a value between 0 and 1. A score of 1.0 indicates that the model correctly predicted the label for every input; a score of 0.0 indicates that no predictions were correct.

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

## Sequence to Sequence

### BLEU

BLEU (Bilingual Evaluation Understudy) measures how closely the model's output matches a reference answer by comparing shared words and short phrases between the two.

Scores range from 0 to 1, where higher values indicate greater similarity to the reference. BLEU works best when the expected output is relatively fixed, such as in translation tasks.

### ROUGE-1

ROUGE-1 measures the word-level overlap between the model's output and the reference answer, balancing how much of the reference is covered and how relevant the output is. A high score indicates that the model's response captures the key content of the reference without excessive irrelevant additions.

Scores range from 0 to 1, where higher values indicate better coverage of the reference content.

### ROUGE-L

ROUGE-L extends ROUGE-1 by also taking word order into account. Rather than counting individual words in isolation, it identifies the longest sequence of words that appear in both the output and the reference in the same order. This makes it more sensitive to whether the model produces a coherent, well-structured response.

Scores range from 0 to 1, and are often somewhat lower than ROUGE-1 because matching both content and order is more difficult. Higher values indicate that the generated output follows the structure of the reference more closely.

## Sequence to Structured Output

### Exact Match

Exact Match is a strict binary metric that checks whether the model's output is completely identical to the reference. A response scores 1 if it matches exactly, and 0 otherwise. It is appropriate when the expected output has only one correct form.

For a full evaluation set, Exact Match is usually reported as the proportion of examples that matched exactly, giving a final score between 0 and 1. A high value means the model reliably produces outputs in the exact required format.

### F1-Score

F1 measures partial correctness by balancing precision (how much of the model's output is correct) and recall (how much of the reference is covered by the output). Unlike Exact Match, it awards partial credit for responses that are mostly correct, making it more informative when outputs can be partially right.

$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Scores range from 0 to 1. A value near 1 means the model is both accurate and complete, while lower values indicate that important content is missing, incorrect, or both.

### Partial Match

Partial Match evaluates the model's output field by field, rather than as a whole. A response that correctly produces some fields but not others receives a score proportional to the number of fields it got right. This is more informative than Exact Match when the structured output contains multiple independent pieces of information.

Scores range from 0 to 1, where higher values indicate that more of the expected fields were generated correctly. For example, a score of 0.75 means that roughly three quarters of the required fields were correct.

### Fuzzy Match

Fuzzy Match is a variant of Partial Match that tolerates minor surface-level differences between the model's output and the reference, such as small spelling variations or punctuation differences. It is useful when the model's answer is semantically correct but does not match the reference character for character.

Scores range from 0 to 1, where higher values indicate greater similarity to the reference despite small formatting or wording differences. A high Fuzzy Match score combined with a lower Exact Match score often means that the model captured the correct content, but not in exactly the required form.

