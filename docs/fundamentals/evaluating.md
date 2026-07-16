# Evaluating

Different metrics can be used to evaluate performances of LLMs, depending highly on the specific task and the domain of the use-case.

## Accuracy

Accuracy measures the proportion of inputs the model classified correctly, expressed as a value between 0 and 1. A score of 1.0 indicates that the model correctly predicted the label for every input; a score of 0.0 indicates that no predictions were correct.

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

## BLEU

BLEU (Bilingual Evaluation Understudy) measures how closely the model's output matches a reference answer by comparing shared words and short phrases between the two.

Scores range from 0 to 1, where higher values indicate greater similarity to the reference. BLEU works best when the expected output is relatively fixed, such as in translation tasks.

## ROUGE-1

ROUGE-1 measures the word-level overlap between the model's output and the reference answer, balancing how much of the reference is covered and how relevant the output is. A high score indicates that the model's response captures the key content of the reference without excessive irrelevant additions.

Scores range from 0 to 1, where higher values indicate better coverage of the reference content.

## ROUGE-L

ROUGE-L extends ROUGE-1 by also taking word order into account. Rather than counting individual words in isolation, it identifies the longest sequence of words that appear in both the output and the reference in the same order. This makes it more sensitive to whether the model produces a coherent, well-structured response.

Scores range from 0 to 1, and are often somewhat lower than ROUGE-1 because matching both content and order is more difficult. Higher values indicate that the generated output follows the structure of the reference more closely.

## Exact Match

Exact Match is a strict binary metric that checks whether the model's output is completely identical to the reference. A response scores 1 if it matches exactly, and 0 otherwise. It is appropriate when the expected output has only one correct form.

## Partial Match

Partial Match evaluates the model's output field by field, rather than as a whole. A response that correctly produces some fields but not others receives a score proportional to the number of fields it got right. This is more informative than Exact Match when the structured output contains multiple independent pieces of information.

Scores range from 0 to 1, where higher values indicate that more of the expected fields were generated correctly. For example, a score of 0.75 means that roughly three quarters of the required fields were correct.

## Fuzzy Match

Fuzzy Match is a variant of Partial Match that tolerates minor surface-level differences between the model's output and the reference, such as small spelling variations or punctuation differences. It is useful when the model's answer is semantically correct but does not match the reference character for character.

Scores range from 0 to 1, where higher values indicate greater similarity to the reference despite small formatting or wording differences. A high Fuzzy Match score combined with a lower Exact Match score often means that the model captured the correct content, but not in exactly the required form.

## Precision

In a classification task, Precision measures how much of the total positive predictions were correct. 

$$\text{Precision} = \frac{\text{true positive predictions}}{\text{true positive predictions + false positive predictions}}$$

## Recall

Recall measures out of all items in a class, how many were correctly identified as part of that class by the model.

$$\text{Recall} = \frac{\text{true positive predictions}}{\text{all items belonging to the class}}$$


## F1-Score

F1 measures partial correctness by balancing precision and recall. Unlike Exact Match, it awards partial credit for responses that are mostly correct, making it more informative when outputs can be partially right.

$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Scores range from 0 to 1. A value near 1 means the model is both accurate and complete, while lower values indicate that important content is missing, incorrect, or both.


## Examples


### Example 1

You want to predict if an e-mail is spam or legitimate. In your dataset 1 out of 100 e-mails are spam. Your trained model predicts "legitmate" for every e-mail. **Accuracy** would be $\frac{99}{100} = 0.99$, making it an unreliable metric in this use-case, where you care about events that are rare. Precision, Recall or the F1-score would be better suited here with all of them being 0.

### Example 2

You want to train a model in the healthcare domain. The model generates a prescription containing "give 500mg" instead of the correct "give 50mg". Text-overlap metrics like ROUGE would score this generation very high because only one character is added in comparison. However, in the healthcare domain this is a fatal error. Evaluating purely based on text-overlap is highly problematic in this case.

### Example 3

You want to extract structured JSON data (name, age, location) from an input text. Your ideal output is ```{"Name": "Martin", "Age": 40, "Location": "Nordrhein-Westfalen"}```. Your model generates ```{"Name": "Martin", "Age": 40, "Location": "North Rhine-Westphalia"}```. Exact Match would score this as a 0, even though the generated output captures everything in the desired output. Partial Match would score this as a $\frac{2}{3} = 0.67$, because the "Name" and "Age" field match exactly. Fuzzy Match would score this higher than Partial Match, because it not only focusses on the exact matchings in the first two fields, but it also reflects the close proximity between the two outputs for the "Location" field, making Fuzzy Match the better suited metric in this example.


## Further Reading

[Huggingface Docs Choosing a Metric↗](https://huggingface.co/docs/evaluate/choosing_a_metric)


[Microsoft AI Playbook List of Evaluation Metrics↗](https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-llms/evaluation/list-of-eval-metrics)


[Weights & Biases Evaluation Metrics↗](https://wandb.ai/onlineinference/genai-research/reports/LLM-evaluation-metrics-A-comprehensive-guide-for-large-language-models--VmlldzoxMjU5ODA4NA)

