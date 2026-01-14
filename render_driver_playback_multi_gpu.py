# render_driver_playback_multi_gpu.py
# Defaults: scene.json + human.json in cwd, Blender on PATH, bpy script is the uploaded one.
# Edit RUN_NUM / RUN_ID as needed, then:  python render_driver_playback_multi_gpu.py

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from multiprocessing import Process
from typing import Iterable

SCENE_JSON = "scene.json"
HUMAN_JSON = "human.json"
INIT_SCRIPT = "init-setting-v1.1.py"
BPY_SCRIPT = "render-script-v5-playback.py"
BLENDER = os.environ.get("BLENDER_PATH", "blender")
TASK_JSON = "tasks_playback.json"

# choose ONE:
RUN_NUM = [-1, 1]     # [num_scenes, num_humans], -1 = all
RUN_ID = [29, 99]     # [scene_id, human_id], -1 = all (set e.g. [4,-1] to render scene 4 with all humans)

DRY_RUN = False
PLAN_ONLY = False
OVERWRITE_TASK_JSON = False


GPU_SELECT_SCRIPT = """\
import bpy
import os

def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

gpu_id = _safe_int(os.environ.get("RENDER_GPU_ID"), default=0)

scene = bpy.context.scene
scene.render.engine = "CYCLES"

addon = bpy.context.preferences.addons.get("cycles")
prefs = getattr(addon, "preferences", None) if addon else None
if prefs:
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    devices = list(getattr(prefs, "devices", []) or [])
    for device in devices:
        device.use = False
    if 0 <= gpu_id < len(devices):
        devices[gpu_id].use = True
    elif devices:
        devices[0].use = True
    for i, device in enumerate(devices):
        print(f"Device[{i}]: {device.name}, Use: {device.use}")
    scene.cycles.device = "GPU"
else:
    print("[WARN] Cycles preferences not found; falling back to default device selection.")
"""


@dataclass(frozen=True)
class RenderTask:
    scene: dict
    human: dict


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


def _gpu_ids_from_env() -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return []
    ids = []
    for part in visible.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


def _gpu_ids_from_nvidia_smi() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    ids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(int(line))
        except ValueError:
            continue
    return ids


def get_available_gpu_ids() -> list[int]:
    env_ids = _gpu_ids_from_env()
    if env_ids:
        return env_ids
    return _gpu_ids_from_nvidia_smi()


def build_tasks(scenes: list[dict], humans: list[dict]) -> list[RenderTask]:
    tasks = []
    for scene in scenes:
        for human in humans:
            tasks.append(RenderTask(scene=scene, human=human))
    return tasks


def write_task_plan(tasks: list[RenderTask], path: str) -> None:
    payload = [{"scene": task.scene, "human": task.human} for task in tasks]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_task_plan(path: str) -> list[RenderTask]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Task plan must be a list of task entries.")
    tasks: list[RenderTask] = []
    for entry in data:
        if not isinstance(entry, dict) or "scene" not in entry or "human" not in entry:
            raise ValueError("Each task entry must include 'scene' and 'human' objects.")
        tasks.append(RenderTask(scene=entry["scene"], human=entry["human"]))
    return tasks


def distribute_tasks(tasks: list[RenderTask], gpu_ids: list[int]) -> dict[int, list[RenderTask]]:
    assignments = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, task in enumerate(tasks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        assignments[gpu_id].append(task)
    return assignments


def _run_task(task: RenderTask, gpu_id: int, gpu_script_path: str) -> None:
    scene = task.scene
    human = task.human

    sid, hid = int(scene["scene_id"]), int(human["human_id"])
    out_base = scene.get("output_path") or scene.get("output_base_path") or "outputs"
    run_dir = os.path.join(out_base, f"{sid:03d}_{hid:03d}")
    meta_path = os.path.join(run_dir, "metadata.json")

    if not os.path.isfile(meta_path):
        print(f"[GPU {gpu_id}] [SKIP] metadata.json not found: {meta_path}")
        return

    cmd = [
        BLENDER,
        "-b",
        scene["scene_blend_path"],
        "-P",
        INIT_SCRIPT,
        "-P",
        gpu_script_path,
        "-P",
        BPY_SCRIPT,
        "--",
        meta_path,
    ]

    env = os.environ.copy()
    env["RENDER_GPU_ID"] = str(gpu_id)

    print(f"[GPU {gpu_id}] scene={sid} human={hid} -> {run_dir}")
    if not DRY_RUN:
        subprocess.run(cmd, env=env, check=False)


def _worker(gpu_id: int, tasks: Iterable[RenderTask], gpu_script_path: str) -> None:
    for task in tasks:
        _run_task(task, gpu_id, gpu_script_path)


def main() -> None:
    scenes = _normalize_items(_load_json(SCENE_JSON), "scene_id", "scene_blend_path")
    humans = _normalize_items(_load_json(HUMAN_JSON), "human_id", "human_blend_path")

    scenes, humans = _select_items(scenes, humans, RUN_NUM, RUN_ID)

    if PLAN_ONLY or OVERWRITE_TASK_JSON or not os.path.exists(TASK_JSON):
        tasks = build_tasks(scenes, humans)
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

    gpu_ids = get_available_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("No GPUs detected. Ensure CUDA_VISIBLE_DEVICES or nvidia-smi is available.")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as gpu_script:
        gpu_script.write(GPU_SELECT_SCRIPT)
        gpu_script_path = gpu_script.name

    assignments = distribute_tasks(tasks, gpu_ids)
    processes: list[Process] = []
    for gpu_id, gpu_tasks in assignments.items():
        proc = Process(target=_worker, args=(gpu_id, gpu_tasks, gpu_script_path))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    try:
        os.remove(gpu_script_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
