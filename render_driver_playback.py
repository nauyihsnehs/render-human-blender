# render_driver_playback.py
# Defaults: scene.json + human.json in cwd, Blender on PATH, bpy script is the uploaded one.
# Edit RUN_NUM / RUN_ID as needed, then:  python render_driver_playback.py

import json
import os
import subprocess

SCENE_JSON = "scene.json"
HUMAN_JSON = "human.json"
INIT_SCRIPT = "init-setting-v1.1.py"
BPY_SCRIPT = "render-script-v5-playback.py"
BLENDER = r"C:\Program Files\Blender Foundation\blender-3.6.18-windows-x64\blender.exe"

# choose ONE:
RUN_NUM = [-1, 1]     # [num_scenes, num_humans], -1 = all
RUN_ID = [29, 99]     # [scene_id, human_id], -1 = all (set e.g. [4,-1] to render scene 4 with all humans)

DRY_RUN = False

scene_data = json.load(open(SCENE_JSON, "r", encoding="utf-8"))
human_data = json.load(open(HUMAN_JSON, "r", encoding="utf-8"))

if isinstance(scene_data, dict):
    scenes = []
    for k, v in scene_data.items():
        if "scene_id" not in v:
            v["scene_id"] = int(k)
        scenes.append(v)
else:
    scenes = scene_data

if isinstance(human_data, dict):
    humans = []
    for k, v in human_data.items():
        if "human_id" not in v:
            v["human_id"] = int(k)
        humans.append(v)
else:
    humans = human_data

scenes = [s for s in scenes if os.path.exists(s.get("scene_blend_path", ""))]
humans = [h for h in humans if os.path.exists(h.get("human_blend_path", ""))]

scenes.sort(key=lambda x: int(x["scene_id"]))
humans.sort(key=lambda x: int(x["human_id"]))

use_id = (RUN_ID[0] != -1) or (RUN_ID[1] != -1)

if use_id:
    if RUN_ID[0] != -1:
        scenes = [s for s in scenes if int(s["scene_id"]) == RUN_ID[0]]
    if RUN_ID[1] != -1:
        humans = [h for h in humans if int(h["human_id"]) == RUN_ID[1]]
else:
    if RUN_NUM[0] != -1:
        scenes = scenes[:RUN_NUM[0]]
    if RUN_NUM[1] != -1:
        humans = humans[:RUN_NUM[1]]

total = len(scenes) * len(humans)
idx = 0

for s in scenes:
    for h in humans:
        idx += 1
        sid, hid = int(s["scene_id"]), int(h["human_id"])

        out_base = s.get("output_path") or s.get("output_base_path") or "outputs"
        run_dir = os.path.join(out_base, f"{sid:03d}_{hid:03d}")
        meta_path = os.path.join(run_dir, "metadata.json")

        if not os.path.isfile(meta_path):
            print(f"[SKIP] metadata.json not found: {meta_path}")
            continue

        cmd = [
            BLENDER,
            "-b",
            s["scene_blend_path"],
            "-P",
            INIT_SCRIPT,
            "-P",
            BPY_SCRIPT,
            "--",
            meta_path,
        ]

        print(f"[{idx}/{total}] scene={sid} human={hid} -> {run_dir}")
        if not DRY_RUN:
            subprocess.run(cmd)
