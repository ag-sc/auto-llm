---
title: Auto-LLM Documentation
layout: default
---

# Auto-LLM Documentation

**AutoLLM** supports you in finding the **right** open source model, architecture and training method for your application. Inspired by "Auto-ML" methods, **AutoLLM** automatically determines the optimal LLM configuration for a problem, trains and evaluates different LLMs for your application. You can choose from different open-source models, training techniques and evaluation metrics.

The platform is part of the project "LLM4KMU".

> Optimierter Einsatz von Open Source Large Language Models (LLMs) in kleinen und mittelständischen Unternehmen (KMUs). Mit Mitteln der Europäischen Union gefördert.

# Start page

## Projects

## Workbenches

# What is Fine-tuning

Fine-tuning aims to increase the performance of 
## Dataset Splits
## Full-weights fine-tuning
## LoRA fine-tuning
# Running an Experiment
## Settings
### Dataset Path

To further fine-tune pre-trained Large Language Models to your specific needs, you first need to provide a dataset. The dataset needs to contain inputs in the form of text, and corresponding ideal answers. During the fine-tuning, the models then get adjusted, so that the likelihood of generating the optimal output given the corresponding input increases.

On the AutoLLM platform you can provide datasets either through paths from [huggingface](https://www.huggingface.co)[^1] 
or your local machine. 

> *Example*: llm-4-kmu/pubmed_mcqa

[^1]: HuggingFace is one of the biggest online platforms in the area of machine learning, providing a vast variety of open datasets.
### Task Category

The Auto-LLM platform supports multiple different tasks. 
#### Sequence to Label

In the Sequence to Label task, the models take in sequences of text and output one label out of multiple possible ones for said data.

A classical example of this task is sentiment analysis of movie reviews, where the models apply either the label "positive" or the label "negative" to a movie review (the input sequence in this case).

>Input: *"That movie was very entertaining!"*
>Output: *Positive* 
#### Sequence to Sequence

In the Sequence to Sequence task, models take in a sequence of text, and output another sequence of text. This is the task most commonly associated with Large Language Models that come in the form of chatbots. Here the users input their sequences, often times a question or a prompt, and in return get a sequence in the form of an answer to their question or prompt.

>Input: "*In which City is the Louvre Museum located?*"
>Output: "*The Louvre Museum is located in Paris, France.*" 

#### Sequence to Structured Output

In the Sequence to Structured Output task, models take in a sequence of text and output structured text in response, often in the form of JSON. Structuring the data makes it easier to process the responses automatically afterwards.

>Input: *"Give me the names of 3 European Prime Ministers"*
>Output: "{
	  "prime_ministers": \[
	    {
	      "country": "France",
	      "prime_minister": "Sébastien Lecornu"
	    },
	    {
	      "country": "Netherlands",
	      "prime_minister": "Rob Jetten"
	    },
	    {
	      "country": "Spain",
	      "prime_minister": "Pedro Sánchez"
	    }
	  ]
	}" 

### Hardware Type & Count

Here you can select the Graphical Processing Units and their counts, that you want to use for your Experiment. Usually, the more powerful a GPU is, the faster the training is completed.
## Models

In the "Models" tab, you then choose which model you want to train. The AutoLLM platform provides you with a pre-selection of models, based on your selection of the task category and hardware parameters. This pre-selection relies on the [huggingface Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/).

The models can be base or instruction fine-tuned models. During instruction fine-tuning, foundation models, that are optimised to predict the next word given an input, learn to adhere to instructions given to them in the input. This is usually what is used to enable "Chatbot"-style interactions with LLMs (e.g. "Give me the names of 3 European Prime Ministers"). 
## Prompts

### Instruction Template
### Input Template
### Output Template
## Validate 

Under the tab "Validate", you can then execute the training that you configured in the previous tabs. 

There are multiple steps that are performed here. Their order is indicated by the "Priority" column:

1. First, the chosen model is evaluated on the task and dataset you picked (see \ref{} for more on the evaluation). 
2. The model is then fine-tuned two times, once full-weight (prefix "fft_") and once using the LoRA method (prefix "lora_"). For more on the fine-tuning procedures see \ref{}.
3. The resulting fine-tuned models are then evaluated again to compare their performance to each other and the baseline

%%Explain everything in the configuration? No, i think..%%
By clicking on the respective run name, you can view and modify the detailed configuration file for each run. The status column tells you if the runs are finished. 

Clicking the eye-symbol next to a run opens up the [Weights&Biases](https://wandb.ai) page of the corresponding step. There you can see all relevant details of the step, including the final evaluation metric score and all generated responses.
## Results
### Evaluation Metrics

Different metrics are used to evaluate the performance of the fine tuned model, depending on the specific task. %%These can be adjusted? Should this be in the docs ?%%

#### Sequence to Label

**Accuracy**

Accuracy measures the proportion of inputs the model classified correctly, expressed as a value between 0 and 1. A score of 1.0 indicates that the model correctly predicted the label for every input; a score of 0.0 indicates that no predictions were correct.

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

#### Sequence to Sequence

**BLEU**

BLEU (Bilingual Evaluation Understudy) measures how closely the model's output matches a reference answer by comparing shared words and short phrases between the two.

Scores range from 0 to 1, where higher values indicate greater similarity to the reference. BLEU works best when the expected output is relatively fixed, such as in translation tasks.

**ROUGE-1**

ROUGE-1 measures the word-level overlap between the model's output and the reference answer, balancing how much of the reference is covered and how relevant the output is. A high score indicates that the model's response captures the key content of the reference without excessive irrelevant additions.

Scores range from 0 to 1, where higher values indicate better coverage of the reference content.

**ROUGE-L**

ROUGE-L extends ROUGE-1 by also taking word order into account. Rather than counting individual words in isolation, it identifies the longest sequence of words that appear in both the output and the reference in the same order. This makes it more sensitive to whether the model produces a coherent, well-structured response.

Scores range from 0 to 1, and are often somewhat lower than ROUGE-1 because matching both content and order is more difficult. Higher values indicate that the generated output follows the structure of the reference more closely.

#### Sequence to Structured Output

**Exact Match**

Exact Match is a strict binary metric that checks whether the model's output is completely identical to the reference. A response scores 1 if it matches exactly, and 0 otherwise. It is appropriate when the expected output has only one correct form.

**F1-Score**

F1 measures partial correctness by balancing precision (how much of the model's output is correct) and recall (how much of the reference is covered by the output). Unlike Exact Match, it awards partial credit for responses that are mostly correct, making it more informative when outputs can be partially right.

$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Scores range from 0 to 1. A value near 1 means the model is both accurate and complete, while lower values indicate that important content is missing, incorrect, or both.

**Partial Match**

Partial Match evaluates the model's output field by field, rather than as a whole. A response that correctly produces some fields but not others receives a score proportional to the number of fields it got right. This is more informative than Exact Match when the structured output contains multiple independent pieces of information. 

Scores range from 0 to 1, where higher values indicate that more of the expected fields were generated correctly. For example, a score of 0.75 means that roughly three quarters of the required fields were correct.

**Fuzzy Match**

Fuzzy Match is a variant of Partial Match that tolerates minor surface-level differences between the model's output and the reference, such as small spelling variations or punctuation differences. It is useful when the model's answer is semantically correct but does not match the reference character for character. 

Scores range from 0 to 1, where higher values indicate greater similarity to the reference despite small formatting or wording differences. A high Fuzzy Match score combined with a lower Exact Match score often means that the model captured the correct content, but not in exactly the required form.




### How to interpret the Results?

- Trainings / Test Split
- Metrics
	- Accuracy
	- F1
What do we mean by Accuracy...


# Monitor

# Chat

