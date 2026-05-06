# What is Fine-tuning

Fine-tuning aims to increase the performance of an existing LLM on a specific task. To do that, different methods can be used. For every experiment that you run, the Auto-LLM platform does two fine-tunings: One full-weights fine-tuning, and a LoRA-based fine-tuning.

## Dataset Splits

For the fine-tuning, the chosen dataset gets split into multiple different, random splits:

- Usually around 60-80% of the dataset gets used during the training. The model predicts an output, given the input in the data, for each items in this **train-split**. The generated outputs are then compared to the ideal outputs, and the internals of the model get adjusted to increase the likelihood of generating the desired output.
- Around 10-20% get withhold to test the settings during the training, this is the **validation-split**.
- The last around 10-20% get withhold to evaluate the final model after all training steps are complete. The model has never seen these items during the training, and so it's performance on this **test-split** is indicative of it's ability to generalise the learned behaviour to future tasks.

## Full-weights fine-tuning

During the Full-weights fine-tuning, all trainable parameters (the weights) that the model uses to generate an output, given an input, are adjusted to increase it's performance on the dataset.

## LoRA fine-tuning
