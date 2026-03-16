import os

from auto_llm.tracker.tracker import WandbClient


Client = WandbClient()
Client.login(key=os.getenv("WANDB_KEY"))
