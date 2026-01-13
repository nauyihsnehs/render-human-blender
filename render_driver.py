# render_driver_min.py
# Defaults: scene.json + human.json in cwd, Blender on PATH, bpy script is the uploaded one.
# Edit RUN_NUM / RUN_ID / EXTRA_OVERRIDES as needed, then:  python render_driver_min.py

import json, os, subprocess

SCENE_JSON = "scene.json"
# SCENE_JSON = "scene-failed.json"
HUMAN_JSON = "human.json"
INIT_SCRIPT = "init-setting-v1.1.py"
BPY_SCRIPT = "render-script-v4.12.py"
# BPY_SCRIPT = "render-sampler-v4.12.py"
BLENDER = r"C:\Program Files\Blender Foundation\blender-3.6.18-windows-x64\blender.exe"

# choose ONE:
RUN_NUM = [-1, 1]     # [num_scenes, num_humans], -1 = all
RUN_ID  = [29, 99]    # [scene_id, human_id], -1 = all (set e.g. [4,-1] to render scene 4 with all humans)

DRY_RUN = False

# any extra keys you want to overwrite in the bpy script:
EXTRA_OVERRIDES = {
    # "N_positions": 16,
}

scene_data = json.load(open(SCENE_JSON, "r", encoding="utf-8"))
human_data = json.load(open(HUMAN_JSON, "r", encoding="utf-8"))

if isinstance(scene_data, dict):
    scenes = []
    for k, v in scene_data.items():
        if "scene_id" not in v: v["scene_id"] = int(k)
        scenes.append(v)
else:
    scenes = scene_data

if isinstance(human_data, dict):
    humans = []
    for k, v in human_data.items():
        if "human_id" not in v: v["human_id"] = int(k)
        humans.append(v)
else:
    humans = human_data

scenes = [s for s in scenes if os.path.exists(s.get("scene_blend_path", ""))]
humans = [h for h in humans if os.path.exists(h.get("human_blend_path", ""))]

scenes.sort(key=lambda x: int(x["scene_id"]))
humans.sort(key=lambda x: int(x["human_id"]))

use_id = (RUN_ID[0] != -1) or (RUN_ID[1] != -1)

if use_id:
    if RUN_ID[0] != -1: scenes = [s for s in scenes if int(s["scene_id"]) == RUN_ID[0]]
    if RUN_ID[1] != -1: humans = [h for h in humans if int(h["human_id"]) == RUN_ID[1]]
else:
    if RUN_NUM[0] != -1: scenes = scenes[:RUN_NUM[0]]
    if RUN_NUM[1] != -1: humans = humans[:RUN_NUM[1]]

total = len(scenes) * len(humans)
idx = 0

for s in scenes:
    for h in humans:
        idx += 1
        sid, hid = int(s["scene_id"]), int(h["human_id"])

        # random seed: {scene_id}{people_id:03d}  (e.g. scene 21 + human 7 => 21007)
        random_seed = int(f"{sid}{hid%1000:03d}")

        out_base = s.get("output_path") or s.get("output_base_path") or "outputs"
        run_dir = os.path.join(out_base, f"{sid:03d}_{hid:03d}")
        os.makedirs(run_dir, exist_ok=True)

        ov = {
            "scene_id": sid,
            "human_id": hid,
            "output_base_path": out_base,
            "human_blend_path": h.get("human_blend_path"),
            "random_seed": random_seed,
        }
        if h.get("human_object_name"): ov["human_object_name"] = h["human_object_name"]
        if s.get("floor_object_names") is not None: ov["floor_object_names"] = s["floor_object_names"]
        if s.get("floor_object_name"): ov["floor_object_name"]  = s["floor_object_name"]
        ov.update(EXTRA_OVERRIDES)

        ov_path = os.path.join(run_dir, "override.json")
        json.dump(ov, open(ov_path, "w", encoding="utf-8"), indent=2)

        cmd = [BLENDER, "-b", s["scene_blend_path"], "-P", INIT_SCRIPT, "-P", BPY_SCRIPT, "--", ov_path]

        print(f"[{idx}/{total}] scene={sid} human={hid} seed={random_seed} -> {run_dir}")
        if not DRY_RUN:
            subprocess.run(cmd)
