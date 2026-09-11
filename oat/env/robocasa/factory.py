"""Sink3 task IDs, matching the existing RoboCasa zarr dataset.

These IDs are canonical RoboCasa IDs, not contiguous indices in this suite.
"""

ROBOCASA_TASK_TO_UID = {
    "TurnOffSinkFaucet": 2,
    "TurnOnSinkFaucet": 4,
    "TurnSinkSpout": 5,
}
ROBOCASA_TASK_NAMES = list(ROBOCASA_TASK_TO_UID)
num_robocasa_tasks = max(ROBOCASA_TASK_TO_UID.values()) + 1
MT_TASKS = {
    "sink3": ["TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout"],
}


def is_multitask(task_name):
    return task_name in MT_TASKS


def get_subtasks(task_name):
    if task_name in MT_TASKS:
        return list(MT_TASKS[task_name])
    get_task_uid(task_name)
    return [task_name]


def get_task_uid(task_name):
    try:
        return ROBOCASA_TASK_TO_UID[task_name]
    except KeyError as exc:
        raise KeyError(f"Unknown RoboCasa task: {task_name}") from exc
