# Run:  python gen_scene_human_json.py
# Output: scene.json, human.json (in current working dir)

from pathlib import Path
import json

SCENE_ROOT = Path(r"E:\evermotion")
HUMAN_ROOT = Path(r"E:\render_people\train")

SCENE_OUT = Path("scene.json")
HUMAN_OUT = Path("human.json")

DEFAULT_FLOOR_OBJECT_NAME = "render_floor_full"
DEFAULT_HUMAN_OBJECT_NAME = "person_render"

# -------- scenes: E:\evermotion\{any}\{any}\{scene}.blend
scene_paths = []
for p in SCENE_ROOT.rglob("*.blend"):
    rel = p.relative_to(SCENE_ROOT)
    if len(rel.parts) == 3:  # folder1/folder2/file.blend
        scene_paths.append(p)

scene_paths = sorted(scene_paths, key=lambda p: (p.parts, p.name))

scenes = []
for i, p in enumerate(scene_paths):
    f1, f2 = p.relative_to(SCENE_ROOT).parts[:2]
    out_dir = SCENE_ROOT / f1 / f2 / "render"
    scenes.append({
        "scene_id": i,
        "scene_blend_path": str(p),
        "output_path": str(out_dir),
        "floor_object_names": [],
        "floor_object_name": DEFAULT_FLOOR_OBJECT_NAME,
    })

json.dump(scenes, open(SCENE_OUT, "w", encoding="utf-8"), indent=2)
print(f"Wrote {SCENE_OUT} ({len(scenes)} scenes)")

# -------- humans: E:\render_people\train\{id:05d}\{id:05d}-3.6.blend
humans = []
for d in sorted(HUMAN_ROOT.glob("[0-9][0-9][0-9][0-9][0-9]"), key=lambda x: int(x.name)):
    pid = int(d.name)
    blend = d / f"{pid:05d}-3.6.blend"
    if blend.exists():
        humans.append({
            "human_id": pid,
            "human_blend_path": str(blend),
            "human_object_name": DEFAULT_HUMAN_OBJECT_NAME,
        })

json.dump(humans, open(HUMAN_OUT, "w", encoding="utf-8"), indent=2)
print(f"Wrote {HUMAN_OUT} ({len(humans)} humans)")
