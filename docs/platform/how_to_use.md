
## Projects

## Workbenches

 Running an Experiment
## Settings
### Dataset Path

To further fine-tune pre-trained Large Language Models to your specific needs, you first need to provide a dataset. The dataset needs to contain inputs in the form of text, and corresponding ideal answers.

On the AutoLLM platform you can provide datasets either through paths from [huggingface](https://www.huggingface.co)
or your local machine.

> *Example*: llm-4-kmu/pubmed_mcqa

HuggingFace is one of the biggest online platforms in the area of machine learning, providing a vast variety of open datasets.

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

By clicking on the respective run name, you can view and modify the detailed configuration file for each run. The status column tells you if the runs are finished.

Clicking the eye-symbol next to a run opens up the [Weights&Biases](https://wandb.ai) page of the corresponding step. There you can see all relevant details of the step, including the final evaluation metric score and all generated responses.


