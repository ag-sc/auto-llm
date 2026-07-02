 The estimator's FLOP-based model overestimates for larger models because it doesn't account for real-world hardware efficiency gains (optimized kernels, memory hierarchy effects, batching).

Larger models emit more CO₂, but not linearly — going from 1B to 14B parameters (~14× more params) only increases emissions ~3-5×, probably because runtime doesn't scale linearly with parameter count for inference/evaluation workloads.

GPU utilization increases with model size — small models (1-2B) use only 25–35% of the GPU, while large models (8–14B) push utilization to 70–88%.

LLaMA 3.2 is the most efficient (1B: 6.1g CO₂, 216s)

Qwen3 is mid-range (1.7B: 11.9g, 454s)

and Gemma 3 is the least efficient (~1B: 11.0g, 458s)


Comparison con le stesse famiglie anche magari tra gemma2, gemma3
In Context Learning

Con il CPU load mode fallback utilizza P=TDP×(0.1+0.9×(cpu_load/100) 
3 ) con TDP = 450w

cpu_load = psutil.cpu_percent(interval=0.5, percpu=False)

the cpu_load reflects the entire node's CPU usage — including other users' jobs if the node isn't exclusively allocated. That's why your force_cpu_power=450 matters: even if the node shows 80% load, some of that may not be yours.

Provare a runnare di nuovo con tracking process

Configurazioni per il train.









### 1. Dataset Selection and Preprocessing
To isolate model architecture as the primary variable for energy consumption, your dataset pipeline must be perfectly rigid.

* **Select a Single Baseline Dataset:** As discussed, use a single dataset like **MedQA (USMLE)** or **MedMCQA** to ensure consistent reasoning complexity and average sequence length across all models.
* **Standardize the Token Count (Crucial):** Do not train for "3 epochs." Because different models use different tokenizers (e.g., Llama's tokenizer splits words differently than Gemma's), 1 epoch of MedQA might be 10 million tokens for Model A but 11 million for Model B. Set your training loop to stop after exactly $X$ million tokens have been processed. If possible we can have this stop_criteria as a parameter in the config
* **Unified Prompt Template:** Wrap every training example in the exact same instruction format (e.g., `Instruction: [Question] \n Options: [A,B,C,D] \n Answer: [Target]`). 


### 3. Model Configuration (PEFT / QLoRA)
Full-parameter fine-tuning is likely too resource-intensive and noisy for this comparison. You should use Parameter-Efficient Fine-Tuning (PEFT).

* **Standardize LoRA Parameters:** Apply the exact same adapter size to all models. For example, fix your LoRA Rank ($r=16$) and Alpha ($\alpha=32$). This ensures you are testing how efficiently the base model integrates new knowledge, rather than testing different adapter sizes.
* **Maximize Batch Size:** To measure true efficiency, you must saturate the GPU. Find the maximum batch size that fits into the GPU memory for *each* model. A model running at 40% memory utilization will yield artificially poor energy efficiency metrics.

### 5. Evaluation and Comparison
Once the models are fine-tuned, you need to map their energy cost against their new capabilities.

* **Standardize Inference:** When testing the models on the test split to get their accuracy scores, use strict, fixed hyperparameters (e.g., `temperature = 0.1`, `top_p = 0.95`).
* **Plot the Pareto Frontier:** Create a scatter plot with **Total Fine-Tuning Energy (Wh)** on the X-axis and **Benchmark Accuracy (%)** on the Y-axis. The models that sit on the top-left edge of this plot represent the optimal trade-off between energy efficiency and learned performance. Add this as a separate function called energy_accuracy_plot in a folder called plots or similar. Add it to evaluation.

