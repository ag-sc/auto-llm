from pydantic import BaseModel, Field
from typing import Optional, Union, List, Any


"""
model: hf
model_args:
  pretrained: Qwen/Qwen3.5-9B
  peft: /vol/auto_llm/sft_models/lora_archaeo_ner_qwen-3.5-9b_conv_zero-shot
  dtype: bfloat16

include_path: tasks
tasks:
  - archaeo_ner

num_fewshot: 0
batch_size: 16
device: cuda:0

apply_chat_template: true

output_path: eval_results
log_samples: true

wandb_args:
  project: german-ner-eval
  entity: llm4kmu
  name: post_archaeo_ner_qwen-3.5-9b_lora
  tags:
    - archaeology
    - archaeo_ner
    - zero_shot
    - fine_tuned
    - lora
"""

class EvaluatorModelArgs(BaseModel):
    
    pretrained: Optional[str] = None
    max_length: Optional[int] = None
    truncation: Optional[bool] = None
    
    dtype: Optional[str]
    peft: Optional[str]
    
    device: str 
    trust_remote_code: bool = False
    parallelize: bool = False
    add_bos_token: Optional[bool] = None


class EvaluatorConfig(BaseModel):

    tasks: Optional[Union[str, List[str]]]

    model_args: EvaluatorModelArgs

    batch_size: int = Field(
        description="Batch Size for evaluation",
        title="Batch Size",
        examples=["1", "16"],
    )
    
    max_batch_size: Optional[int]

    limit: Optional[float]
    output_path: Optional[str]

    apply_chat_template: Union[bool, str] = False
    write_out: bool = True
    log_samples: bool = True

