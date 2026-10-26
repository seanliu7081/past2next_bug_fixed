"""DexMimicGen dexterous-hand tasks: env names, embodiments, cameras, horizons and stable task ids.

Pure metadata (no simulator import) shared by the HDF5 -> zarr converter, task configs and the
runner. Horizons follow DexMimicGen's generate_training_config.py; the dataset's own env_args
(stored in each zarr's root attrs) remain the source of truth for constructing environments.
"""

# Arm-level observation prefixes per embodiment; robot0 / robot0_right is the right arm.
ARM_PREFIXES = {
    "panda_dex": ("robot0", "robot1"),
    "gr1": ("robot0_right", "robot0_left"),
}

DEXMG_TASKS = {
    "box_cleanup": dict(
        env_name="TwoArmBoxCleanup", hdf5="two_arm_box_cleanup.hdf5", embodiment="panda_dex",
        uid=0, horizon=400, cameras=("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand")),
    "drawer_cleanup": dict(
        env_name="TwoArmDrawerCleanup", hdf5="two_arm_drawer_cleanup.hdf5", embodiment="panda_dex",
        uid=1, horizon=550, cameras=("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand")),
    "lift_tray": dict(
        env_name="TwoArmLiftTray", hdf5="two_arm_lift_tray.hdf5", embodiment="panda_dex",
        uid=2, horizon=750, cameras=("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand")),
    "coffee": dict(
        env_name="TwoArmCoffee", hdf5="two_arm_coffee.hdf5", embodiment="gr1",
        uid=3, horizon=400, cameras=("agentview", "robot0_eye_in_left_hand", "robot0_eye_in_right_hand")),
    "pouring": dict(
        env_name="TwoArmPouring", hdf5="two_arm_pouring.hdf5", embodiment="gr1",
        uid=4, horizon=400, cameras=("agentview", "robot0_eye_in_left_hand", "robot0_eye_in_right_hand")),
    # The HDF5 env_args name TwoArmCanSortBlue although the demos mix colours.
    "can_sort": dict(
        env_name="TwoArmCanSortRandom", hdf5="two_arm_can_sort_random.hdf5", embodiment="gr1",
        uid=5, horizon=400, cameras=("frontview", "robot0_eye_in_left_hand", "robot0_eye_in_right_hand")),
}


def get_task(task_name):
    try:
        return dict(DEXMG_TASKS[task_name], name=task_name)
    except KeyError as exc:
        raise KeyError(f"Unknown DexMimicGen task {task_name!r}; known: {sorted(DEXMG_TASKS)}") from exc


def get_task_uid(task_name):
    return get_task(task_name)["uid"]


def get_subtasks(task_name):
    """Single-task suites: the runner's task pattern is the task itself."""
    get_task(task_name)
    return [task_name]
