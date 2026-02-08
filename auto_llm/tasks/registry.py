from auto_llm.tasks.sequence_to_label_task import SequenceToLabelTask
from auto_llm.tasks.sequence_to_sequence_task import SequenceToSequenceTask
from auto_llm.tasks.sequence_to_structured_output_task import SequenceToStructuredOutputTask

TASKS = {
    SequenceToSequenceTask.name: SequenceToSequenceTask,
    SequenceToLabelTask.name: SequenceToLabelTask,
    SequenceToStructuredOutputTask.name: SequenceToStructuredOutputTask
}