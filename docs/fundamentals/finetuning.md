# Fine-tuning

Fine-tuning aims to increase the performance of an existing LLM on a specific task. To do that, different methods can be used. For every experiment that you run, the Auto-LLM platform does two fine-tunings: One full-weights fine-tuning, and a parameter efficient fine-tuning.

## Full-weights Fine-Tuning

During the Full-weights fine-tuning, all trainable parameters (the weights) that the model uses to generate an output, given an input, are adjusted to increase it's performance on the dataset. As modern LLMs usually have billions of parameters, storing and updating all of them requires siginificant memory and compute power.

## Parameter Efficient Fine-Tuning

To remedy for the time- and resource-intensiveness of full-weights fine-tuning, parameter efficient fine-tuning (PEFT) was created. PEFT keeps most of the model frozen and only trains a small set of added parameters.

### LoRA

The most popular PEFT method is Low-Rank Adaptation (LoRA). Rather than touching the model's existing weights, LoRA attaches small trainable modules to specific layers. During training, only these modules are updated — the rest of the model stays exactly as it was.

A single setting called the rank controls how large these modules are. Lower rank means fewer parameters to train, which is faster and works better when your dataset is small. Higher rank gives the model more room to adapt, but requires more memory.

In practice, LoRA trains less than 1% of the total parameters, produces no slowdown at inference time (the modules can be merged back into the model after training), and tends to generalise better on small datasets than full fine-tuning.

The Auto-LLM platform runs both approaches side by side so you can see which one actually performs better on your data.

## Further Reading

[Databricks Blog on Finetuning↗](https://www.databricks.com/blog/llm-fine-tuning)


[Google Cloud Blog on Finetuning↗](https://cloud.google.com/use-cases/fine-tuning-ai-models)
