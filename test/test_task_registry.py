from auto_llm.tasks.task import TaskRegistry


def test_task_registry():
    TaskRegistry.setup_tasks()

    print(TaskRegistry.get_task_names())

    task = TaskRegistry.get_task("sequence_to_label")
    print(task.description)
