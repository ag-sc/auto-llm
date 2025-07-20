import os

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from auto_llm.builder.data_builder import DataBuilder
from auto_llm.pre_processor.sft_pre_procesor import SftPreProcessor

model_name = "google/gemma-2-2b-it"
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
output_dir = ".cache/"
dataset_path = "data/train_instruct_model_wo_sys.jsonl"
# dataset_path = "data/train_base_model.jsonl"


builder = DataBuilder(dataset_path=dataset_path)
ds_dict = builder.build()

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    token=os.getenv("HF_TOKEN"),
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name,
    token=os.getenv("HF_TOKEN"),
)

pre_processor = SftPreProcessor(tokenizer=tokenizer, completion_only_loss=False)  # True

ds_dict = ds_dict.map(
    function=pre_processor.pre_process,
    fn_kwargs=dict(
        max_length=256,
        truncation=True,
    ),
    batched=True,
)


training_args = SFTConfig(
    max_length=512,
    output_dir=output_dir,
    # completion_only_loss=True,
    # assistant_only_loss=True,
    report_to="wandb",
    run_name="sft-test",
    logging_steps=1,
    num_train_epochs=10,
    # TODO: skip dataset prep for Conversational DS with assistant_only_loss=True
    dataset_kwargs={"skip_prepare_dataset": True},
)


os.environ["WANDB_PROJECT"] = "llm4kmu-train"


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # target_modules="all-linear",
    # modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    peft_config=peft_config,
    train_dataset=ds_dict["train"],
    eval_dataset=ds_dict["val"],
    # optimizers=...,
)

trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
