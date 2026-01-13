# render-playback-v3.py
# Run:
#   blender your_scene.blend --python render-playback-v3.py -- /path/to/metadata.json

import bpy
import json
import os
import sys
from mathutils import Matrix

JSON_PATH = r"E:\evermotion\Archinteriors Vol.48-3.6\AI48_blender__001\render\000_000\metadata.json"


def bu_from_m(val_m: float, SU: float) -> float:
    return float(val_m) / float(SU if SU > 0 else 1.0)


def parse_metadata_path_from_argv() -> str:
    argv = sys.argv
    if "--" in argv:
        i = argv.index("--") + 1
        if i < len(argv):
            return argv[i]
    if os.path.isfile(JSON_PATH):
        return JSON_PATH
    blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else ""
    return os.path.join(blend_dir, "metadata.json")


def load_json(path: str) -> dict:
    path = bpy.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"metadata.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_people_one_object(blend_path: str, object_name: str = ""):
    blend_path = bpy.path.abspath(blend_path)
    if not os.path.isfile(blend_path):
        raise FileNotFoundError(
            f"Human blend not found: {blend_path}\n"
            f"(metadata.human_blend_path must exist on THIS machine)"
        )

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        names = list(data_from.objects)
        if not names:
            raise RuntimeError("No objects in human.blend")
        chosen = object_name if (object_name and object_name in names) else names[0]
        data_to.objects = [chosen]

    obj = bpy.data.objects.get(chosen)
    if obj is None:
        raise RuntimeError(f"Failed to load object: {chosen}")

    if obj.name not in bpy.context.scene.objects:
        bpy.context.scene.collection.objects.link(obj)

    return obj


def ensure_camera_named(cam_name: str):
    scene = bpy.context.scene

    obj = bpy.data.objects.get(cam_name)
    if obj and obj.type == "CAMERA":
        scene.camera = obj
        return obj

    if scene.camera and scene.camera.type == "CAMERA":
        scene.camera.name = cam_name
        scene.camera.data.name = cam_name
        return scene.camera

    cam_data = bpy.data.cameras.new(cam_name)
    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def ensure_disk_area_light():
    name = "PeopleDiskLight"
    obj = bpy.data.objects.get(name)
    if obj and obj.type == "LIGHT" and obj.data.type == "AREA":
        obj.data.shape = "DISK"
        return obj

    ld = bpy.data.lights.new(name, type="AREA")
    ld.shape = "DISK"
    obj = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def clear_animation(thing):
    if thing is None:
        return
    try:
        thing.animation_data_clear()
    except Exception:
        pass


def keyframe_transform(obj, frame: int):
    obj.keyframe_insert(data_path="location", frame=frame)
    if obj.rotation_mode != "QUATERNION":
        obj.rotation_mode = "QUATERNION"
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)


def find_cam_target_like_render(cam_obj, preferred_target_name: str | None = None):
    # Match render-script-v4.12.py:
    # 1) bpy.data.objects.get(f"{cam_obj.name}.Target")
    # 2) else first TRACK_TO constraint target
    if preferred_target_name:
        t = bpy.data.objects.get(preferred_target_name)
        if t is not None:
            return t

    t = bpy.data.objects.get(f"{cam_obj.name}.Target")
    if t is None:
        for cns in getattr(cam_obj, "constraints", []):
            if cns.type == "TRACK_TO" and getattr(cns, "target", None) is not None:
                t = cns.target
                break
    return t


def main():
    meta_path = parse_metadata_path_from_argv()
    meta = load_json(meta_path)

    renders = list(meta.get("renders", []))
    if not renders:
        raise RuntimeError("metadata.json has no 'renders' entries.")

    SU = float(meta.get("scene_unit_scale_length", 1.0))

    # --- Load / append human (same as render) ---
    human_blend = meta.get("human_blend_path")
    human_obj_name = meta.get("human_object_name", "")
    if not human_blend:
        raise RuntimeError("metadata.json missing 'human_blend_path'.")

    people_obj = bpy.data.objects.get(human_obj_name) if human_obj_name else None
    if people_obj is None:
        people_obj = append_people_one_object(human_blend, human_obj_name)

    if "human_pass_index" in meta:
        try:
            people_obj.pass_index = int(meta["human_pass_index"])
        except Exception:
            pass

    # --- Camera / target (DO NOT create/modify constraints) ---
    cameras = meta.get("cameras", [])
    cams_by_id = {int(c["camera_id"]): c for c in cameras if "camera_id" in c}

    cam_name = cameras[0].get("camera_name", "Camera") if cameras else "Camera"
    cam_obj = ensure_camera_named(cam_name)
    cam_obj.rotation_mode = "QUATERNION"
    people_obj.rotation_mode = "QUATERNION"

    # Apply recorded intrinsics (constant) if present (stored by render-script-v4.12)
    intr = cameras[0].get("intrinsics") if cameras else None
    if isinstance(intr, dict):
        if intr.get("lens_unit") is not None and hasattr(cam_obj.data, "lens_unit"):
            try:
                cam_obj.data.lens_unit = str(intr["lens_unit"])
            except Exception:
                pass
        if intr.get("sensor_width") is not None and hasattr(cam_obj.data, "sensor_width"):
            try:
                cam_obj.data.sensor_width = float(intr["sensor_width"])
            except Exception:
                pass
        if intr.get("sensor_fit") is not None and hasattr(cam_obj.data, "sensor_fit"):
            try:
                cam_obj.data.sensor_fit = str(intr["sensor_fit"])
            except Exception:
                pass
        if intr.get("shift_x") is not None and hasattr(cam_obj.data, "shift_x"):
            try:
                cam_obj.data.shift_x = float(intr["shift_x"])
            except Exception:
                pass
        if intr.get("shift_y") is not None and hasattr(cam_obj.data, "shift_y"):
            try:
                cam_obj.data.shift_y = float(intr["shift_y"])
            except Exception:
                pass


    # --- Camera target (match render-script-v4.12) ---
    # render-script-v4.12 expects a target object (usually f"{Camera}.Target") and sets it to the person's origin.
    target_names = [c.get("target_name") for c in cameras if c.get("target_name")]
    preferred_target_name = target_names[0] if target_names else None
    cam_target = find_cam_target_like_render(cam_obj, preferred_target_name)

    if preferred_target_name and cam_target is None:
        raise RuntimeError(
            "Metadata includes a camera target_name, but the target object was not found.\n"
            f"Expected an object named '{preferred_target_name}' (or '{cam_obj.name}.Target') "
            "or a TRACK_TO constraint with a target on the camera."
        )

    # --- Light ---
    light_obj = ensure_disk_area_light()
    light_obj.rotation_mode = "QUATERNION"

    # Clear animation (match render script intent)
    clear_animation(people_obj)
    clear_animation(cam_obj)
    clear_animation(cam_obj.data)
    clear_animation(light_obj)
    clear_animation(light_obj.data)
    if cam_target:
        clear_animation(cam_target)

    # Lookup people samples
    people_samples = meta.get("people_samples", [])
    people_by_key = {
        (int(p["pos_id"]), int(p["yaw_local"])): p
        for p in people_samples
        if "pos_id" in p and "yaw_local" in p
    }

    # Timeline range
    max_frame = max(int(r["frame"]) for r in renders)
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = max_frame

    # Keyframe by render frames
    for r in sorted(renders, key=lambda x: int(x["frame"])):
        frame = int(r["frame"])
        pos_id = int(r["pos_id"])
        yaw_local = int(r["yaw_local"])
        camera_id = int(r["camera_id"])
        mode = r.get("mode", "")

        scene.frame_set(frame)

        # --- People: set matrix_world directly (no 1-frame lag) ---
        p = people_by_key.get((pos_id, yaw_local))
        if not p or "matrix_world" not in p:
            raise RuntimeError(f"Missing people_samples matrix_world for pos_id={pos_id}, yaw_local={yaw_local}")

        people_M = Matrix(p["matrix_world"])
        people_obj.matrix_world = people_M
        people_origin = people_M.to_translation()
        keyframe_transform(people_obj, frame)

        # --- Camera ---
        c = cams_by_id.get(camera_id)
        if not c or "matrix_world" not in c:
            raise RuntimeError(f"Missing cameras matrix_world for camera_id={camera_id}")

        cam_M = Matrix(c["matrix_world"])
        cam_obj.matrix_world = cam_M
        keyframe_transform(cam_obj, frame)

        if c.get("focal_mm") is not None:
            cam_obj.data.lens = float(c["focal_mm"])
            cam_obj.data.keyframe_insert(data_path="lens", frame=frame)

        # --- Target: follow person on every frame (render-script-v4.12 sets target_obj.location every render step) ---
        if cam_target:
            cam_target.location = people_origin
            cam_target.keyframe_insert(data_path="location", frame=frame)

        # --- Light ---
        if mode == "no_light":
            light_obj.data.energy = 0.0
        elif mode == "lit":
            lrec = r.get("light") or {}
            if "matrix_world" in lrec:
                light_obj.matrix_world = Matrix(lrec["matrix_world"])
                keyframe_transform(light_obj, frame)

            if lrec.get("power") is not None:
                light_obj.data.energy = float(lrec["power"])
            if lrec.get("size_bu") is not None:
                light_obj.data.size = float(lrec["size_bu"])
            elif lrec.get("size_m") is not None:
                size_m = float(lrec["size_m"])
                # Usually meters (<= 1). Convert using scene_unit_scale_length; if it looks big, assume it's already BU.
                light_obj.data.size = float(bu_from_m(size_m, SU)) if size_m <= 1.0 else size_m
            if lrec.get("color") is not None:
                c3 = lrec["color"]
                light_obj.data.color = (float(c3[0]), float(c3[1]), float(c3[2]))

        light_obj.data.keyframe_insert(data_path="energy", frame=frame)
        if hasattr(light_obj.data, "size"):
            light_obj.data.keyframe_insert(data_path="size", frame=frame)
        light_obj.data.keyframe_insert(data_path="color", frame=frame)

    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()
    print(f"[PLAYBACK] Done. Keyframed {len(renders)} frames from: {bpy.path.abspath(meta_path)}")


if __name__ == "__main__":
    main()