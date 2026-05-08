# Fine-tuning

Fine-tuning aims to increase the performance of an existing LLM on a specific task. To do that, different methods can be used. For every experiment that you run, the Auto-LLM platform does two fine-tunings: One full-weights fine-tuning, and a parameter efficient fine-tuning.

## Dataset Splits

For the fine-tuning, the chosen [dataset](../datasets) gets split into multiple different, random splits:

- Usually around 60-80% of the dataset gets used during the training. The model predicts an output, given the input in the data, for each items in this **train-split**. The generated outputs are then compared to the ideal outputs, and the internals of the model get adjusted to increase the likelihood of generating the desired output.
- Around 10-20% get withhold to test the settings during the training, this is the **validation-split**.
- The last around 10-20% get withhold to evaluate the final model after all training steps are complete. The model has never seen these items during the training, and so it's performance on this **test-split** is indicative of it's ability to generalise the learned behaviour to future tasks.

## Full-weights Fine-Tuning

During the Full-weights fine-tuning, all trainable parameters (the weights) that the model uses to generate an output, given an input, are adjusted to increase it's performance on the dataset. As modern LLMs usually have billions of parameters, storing and updating all of them requires siginificant memory and compute power.

## Parameter Efficient Fine-Tuning

To remedy for the time- and resource-intensiveness of full-weights fine-tuning, parameter efficient fine-tuning (PEFT) was created. PEFT keeps most of the model frozen and only trains a small set of added parameters.

### LoRA

The most popular PEFT method is Low-Rank Adaptation (LoRA). Rather than touching the model's existing weights, LoRA attaches small trainable modules to specific layers. During training, only these modules are updated — the rest of the model stays exactly as it was.

A single setting called the rank controls how large these modules are. Lower rank means fewer parameters to train, which is faster and works better when your dataset is small. Higher rank gives the model more room to adapt, but requires more memory.

In practice, LoRA trains less than 1% of the total parameters, produces no slowdown at inference time (the modules can be merged back into the model after training), and tends to generalise better on small datasets than full fine-tuning.

The Auto-LLM platform runs both approaches side by side so you can see which one actually performs better on your data.


