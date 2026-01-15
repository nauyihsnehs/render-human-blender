# render_driver_multi_threading.py
# Defaults: scene.json + human.json in cwd, Blender on PATH, bpy script is the uploaded one.
# Edit RUN_NUM / RUN_ID / THREAD_COUNT as needed, then:  python render_driver_multi_threading.py

import json
import os
import subprocess
from dataclasses import dataclass
from threading import Thread
from typing import Iterable

# SCENE_JSON = "scene.json"
SCENE_JSON = "scene-failed.json"
HUMAN_JSON = "human.json"
INIT_SCRIPT = "init-setting-v1.1.py"
BPY_SCRIPT = "render-script-v5.1-sample.py"
BLENDER = "/mnt/data2/ssy/blender-3.6.23-linux-x64/blender"
TASK_JSON = "tasks_sampling.json"

RUN_NUM = [-1, -1]  # [num_scenes, num_humans], -1 = all
RUN_ID = [-1, -1]  # [scene_id, human_id], -1 = all (set e.g. [4,-1] to render scene 4 with all humans)

DRY_RUN = False
PLAN_ONLY = False
OVERWRITE_TASK_JSON = False

THREAD_COUNT = 4

# any extra keys you want to overwrite in the bpy script:
EXTRA_OVERRIDES = {
    # "N_positions": 16,
}


@dataclass(frozen=True)
class RenderTask:
    scene_id: int
    human_id: int
    thread_id: int


def _load_json(path: str) -> list[dict]:
    data = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(data, dict):
        items = []
        for key, value in data.items():
            if "scene_id" not in value and "human_id" not in value:
                value["scene_id" if "scene_blend_path" in value else "human_id"] = int(key)
            items.append(value)
        return items
    return data


def _normalize_items(items: list[dict], id_key: str, path_key: str) -> list[dict]:
    normalized = []
    for item in items:
        if id_key not in item:
            item[id_key] = int(item.get("id", 0))
        if os.path.exists(item.get(path_key, "")):
            normalized.append(item)
    normalized.sort(key=lambda x: int(x[id_key]))
    return normalized


def _select_items(
        scenes: list[dict],
        humans: list[dict],
        run_num: list[int],
        run_id: list[int],
) -> tuple[list[dict], list[dict]]:
    use_id = (run_id[0] != -1) or (run_id[1] != -1)
    if use_id:
        if run_id[0] != -1:
            scenes = [s for s in scenes if int(s["scene_id"]) == run_id[0]]
        if run_id[1] != -1:
            humans = [h for h in humans if int(h["human_id"]) == run_id[1]]
    else:
        if run_num[0] != -1:
            scenes = scenes[: run_num[0]]
        if run_num[1] != -1:
            humans = humans[: run_num[1]]
    return scenes, humans


def get_thread_ids() -> list[int]:
    count = max(1, int(THREAD_COUNT))
    return list(range(count))


def build_tasks(scenes: list[dict], humans: list[dict], thread_ids: list[int]) -> list[RenderTask]:
    tasks = []
    idx = 0
    for scene in scenes:
        for human in humans:
            thread_id = thread_ids[idx % len(thread_ids)]
            tasks.append(
                RenderTask(
                    scene_id=int(scene["scene_id"]),
                    human_id=int(human["human_id"]),
                    thread_id=thread_id,
                )
            )
            idx += 1
    return tasks


def write_task_plan(tasks: list[RenderTask], path: str) -> None:
    payload = [
        {"scene_id": task.scene_id, "human_id": task.human_id, "thread_id": task.thread_id}
        for task in tasks
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_task_plan(path: str) -> list[RenderTask]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Task plan must be a list of task entries.")
    tasks: list[RenderTask] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each task entry must include scene_id, human_id, and thread_id fields.")
        if {"scene_id", "human_id", "thread_id"} - set(entry.keys()):
            raise ValueError("Each task entry must include scene_id, human_id, and thread_id fields.")
        tasks.append(
            RenderTask(
                scene_id=int(entry["scene_id"]),
                human_id=int(entry["human_id"]),
                thread_id=int(entry["thread_id"]),
            )
        )
    return tasks


def _index_by_id(items: list[dict], id_key: str) -> dict[int, dict]:
    return {int(item[id_key]): item for item in items}


def _run_task(task: RenderTask, scenes_by_id: dict[int, dict], humans_by_id: dict[int, dict]) -> None:
    scene = scenes_by_id.get(task.scene_id)
    human = humans_by_id.get(task.human_id)
    if scene is None or human is None:
        print(
            f"[Thread {task.thread_id}] [SKIP] Missing scene/human for scene={task.scene_id} human={task.human_id}"
        )
        return

    sid, hid = int(scene["scene_id"]), int(human["human_id"])
    random_seed = int(f"{sid}{hid % 1000:03d}")

    out_base = scene.get("output_path") or scene.get("output_base_path") or "outputs"
    run_dir = os.path.join(out_base, f"{sid:03d}_{hid:03d}")
    os.makedirs(run_dir, exist_ok=True)

    overrides = {
        "scene_id": sid,
        "human_id": hid,
        "output_base_path": out_base,
        "human_blend_path": human.get("human_blend_path"),
        "random_seed": random_seed,
    }
    if human.get("human_object_name"):
        overrides["human_object_name"] = human["human_object_name"]
    if scene.get("floor_object_names") is not None:
        overrides["floor_object_names"] = scene["floor_object_names"]
    if scene.get("floor_object_name"):
        overrides["floor_object_name"] = scene["floor_object_name"]
    overrides.update(EXTRA_OVERRIDES)

    ov_path = os.path.join(run_dir, "override.json")
    with open(ov_path, "w", encoding="utf-8") as handle:
        json.dump(overrides, handle, indent=2)

    cmd = [
        BLENDER,
        "-b",
        scene["scene_blend_path"],
        "-P",
        INIT_SCRIPT,
        "-P",
        BPY_SCRIPT,
        "--",
        ov_path,
    ]

    print(f"[Thread {task.thread_id}] scene={sid} human={hid} seed={random_seed} -> {run_dir}")
    if not DRY_RUN:
        subprocess.run(cmd, check=False)


def _worker(
        tasks: Iterable[RenderTask],
        scenes_by_id: dict[int, dict],
        humans_by_id: dict[int, dict],
) -> None:
    for task in tasks:
        _run_task(task, scenes_by_id, humans_by_id)


def _group_tasks_by_thread(tasks: Iterable[RenderTask]) -> dict[int, list[RenderTask]]:
    assignments: dict[int, list[RenderTask]] = {}
    for task in tasks:
        assignments.setdefault(task.thread_id, []).append(task)
    return assignments


def main() -> None:
    scenes = _normalize_items(_load_json(SCENE_JSON), "scene_id", "scene_blend_path")
    humans = _normalize_items(_load_json(HUMAN_JSON), "human_id", "human_blend_path")

    scenes, humans = _select_items(scenes, humans, RUN_NUM, RUN_ID)

    if PLAN_ONLY or OVERWRITE_TASK_JSON or not os.path.exists(TASK_JSON):
        thread_ids = get_thread_ids()
        tasks = build_tasks(scenes, humans, thread_ids)
        if not tasks:
            print("No render tasks found.")
            return
        write_task_plan(tasks, TASK_JSON)
        print(f"Task plan written: {TASK_JSON}")
        if PLAN_ONLY:
            return

    tasks = load_task_plan(TASK_JSON)
    if not tasks:
        print("No render tasks found in task plan.")
        return

    scenes_by_id = _index_by_id(scenes, "scene_id")
    humans_by_id = _index_by_id(humans, "human_id")

    assignments = _group_tasks_by_thread(tasks)
    threads: list[Thread] = []
    for thread_id, thread_tasks in assignments.items():
        thread = Thread(
            target=_worker,
            args=(thread_tasks, scenes_by_id, humans_by_id),
            daemon=False,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
