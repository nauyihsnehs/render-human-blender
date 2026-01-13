# render-script-v5-playback.py
# Run:
#   blender your_scene.blend --python render-script-v5-playback.py -- /path/to/metadata.json

import bpy
import json
import os
import sys
from mathutils import Matrix


def parse_metadata_path_from_argv(default_path: str | None = None) -> str:
    argv = sys.argv
    if "--" in argv:
        i = argv.index("--") + 1
        if i < len(argv):
            return argv[i]
    if default_path and os.path.isfile(default_path):
        return default_path
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


def find_cam_target_like_render(cam_obj, preferred_target_name: str | None = None):
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


def apply_intrinsics(cam_obj, intr: dict | None):
    if not isinstance(intr, dict):
        return
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


# ============================================================
# Compositor (copied from render-script-v4.12)
# ============================================================

def ensure_compositor_nodes(human_pass_index: int, rgb_format: str = "PNG"):
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nodes = nt.nodes
    links = nt.links

    scene.view_layers[0].use_pass_object_index = True

    if hasattr(scene.view_layers[0], "use_pass_diffuse_color"):
        scene.view_layers[0].use_pass_diffuse_color = True

    for n in list(nodes):
        if (n.type == "OUTPUT_FILE" or n.type == "ID_MASK") and n.name.startswith("PR_"):
            nodes.remove(n)

    rl = nodes.get("Render Layers")
    if rl is None:
        rl = nodes.new("CompositorNodeRLayers")

    out_rgb = nodes.new("CompositorNodeOutputFile")
    out_rgb.name = "PR_RGB"
    out_rgb.base_path = ""
    fmt = str(rgb_format or "PNG").upper()
    if fmt in ("EXR", "OPEN_EXR", "OPENEXR"):
        out_rgb.format.file_format = "OPEN_EXR"
        out_rgb.format.color_mode = "RGB"
        out_rgb.format.color_depth = "16"
        if hasattr(out_rgb.format, "exr_codec"):
            out_rgb.format.exr_codec = "ZIP"
    else:
        out_rgb.format.file_format = "PNG"
        out_rgb.format.color_mode = "RGB"
        out_rgb.format.color_depth = "8"
    out_rgb.file_slots.clear()
    out_rgb.file_slots.new("Image")
    out_rgb.file_slots[-1].path = "tmp_rgb_####"

    out_dat = nodes.new("CompositorNodeOutputFile")
    out_dat.name = "PR_DATA"
    out_dat.base_path = ""
    out_dat.format.file_format = "OPEN_EXR"
    out_dat.format.color_mode = "RGB"
    out_dat.format.color_depth = "32"
    if hasattr(out_dat.format, "exr_codec"):
        out_dat.format.exr_codec = "ZIP"
    out_dat.file_slots.clear()
    out_dat.file_slots.new("Depth")
    out_dat.file_slots[-1].path = "tmp_dpt_####"
    out_dat.file_slots.new("Normal")
    out_dat.file_slots[-1].path = "tmp_nor_####"

    out_alb = nodes.new("CompositorNodeOutputFile")
    out_alb.name = "PR_ALB"
    out_alb.base_path = ""
    out_alb.format.file_format = "PNG"
    out_alb.format.color_mode = "RGB"
    out_alb.format.color_depth = "8"
    out_alb.file_slots.clear()
    out_alb.file_slots.new("Denoising Albedo")
    out_alb.file_slots[-1].path = "tmp_alb_####"

    out_dif = nodes.new("CompositorNodeOutputFile")
    out_dif.name = "PR_DIF"
    out_dif.base_path = ""
    out_dif.format.file_format = "PNG"
    out_dif.format.color_mode = "RGB"
    out_dif.format.color_depth = "8"
    out_dif.file_slots.clear()
    out_dif.file_slots.new("Diffuse Color")
    out_dif.file_slots[-1].path = "tmp_dif_####"

    idm = nodes.new("CompositorNodeIDMask")
    idm.name = "PR_IDMASK"
    idm.index = int(human_pass_index)
    if hasattr(idm, "use_antialiasing"):
        idm.use_antialiasing = True

    out_msk = nodes.new("CompositorNodeOutputFile")
    out_msk.name = "PR_MSK"
    out_msk.base_path = ""
    out_msk.format.file_format = "PNG"
    out_msk.format.color_mode = "BW"
    out_msk.format.color_depth = "8"
    out_msk.file_slots.clear()
    out_msk.file_slots.new("Mask")
    out_msk.file_slots[-1].path = "tmp_msk_####"

    def safe_link(out_name, node, in_name):
        s = rl.outputs.get(out_name)
        i = node.inputs.get(in_name)
        if s and i:
            links.new(s, i)

    safe_link("Image", out_rgb, "Image")
    safe_link("Depth", out_dat, "Depth")
    safe_link("Normal", out_dat, "Normal")
    safe_link("Denoising Albedo", out_alb, "Denoising Albedo")
    safe_link("DiffCol", out_dif, "Diffuse Color")
    safe_link("Diffuse Color", out_dif, "Diffuse Color")

    sock_idx = rl.outputs.get("IndexOB")
    if sock_idx and idm.inputs.get("ID value"):
        links.new(sock_idx, idm.inputs["ID value"])

    if idm.outputs.get("Alpha") and out_msk.inputs.get("Mask"):
        links.new(idm.outputs["Alpha"], out_msk.inputs["Mask"])

    return out_rgb, out_dat, out_alb, out_dif, out_msk, idm


def render_and_rename(run_dir, frame_idx, camera_id, light_id, write_data_once, write_mask_once):
    scene = bpy.context.scene
    out_rgb = scene.node_tree.nodes.get("PR_RGB")
    out_dat = scene.node_tree.nodes.get("PR_DATA")
    out_alb = scene.node_tree.nodes.get("PR_ALB")
    out_dif = scene.node_tree.nodes.get("PR_DIF")
    out_msk = scene.node_tree.nodes.get("PR_MSK")

    for node in [out_rgb, out_dat, out_alb, out_dif, out_msk]:
        node.base_path = run_dir

    out_dat.mute = not write_data_once
    out_alb.mute = not write_data_once
    out_dif.mute = not write_data_once
    out_msk.mute = not write_mask_once

    scene.node_tree.update_tag()
    bpy.context.view_layer.update()

    scene.frame_set(frame_idx)
    bpy.context.view_layer.update()

    bpy.ops.render.render(write_still=False, use_viewport=False)

    frame_str = f"{frame_idx:04d}"

    rgb_ext = ".exr" if (out_rgb and getattr(out_rgb.format, "file_format", "") == "OPEN_EXR") else ".png"
    tmp_rgb = os.path.join(run_dir, f"tmp_rgb_{frame_str}{rgb_ext}")
    dst_rgb = os.path.join(run_dir, f"{camera_id:03d}_{light_id:03d}_rgb{rgb_ext}")
    if os.path.isfile(tmp_rgb):
        if os.path.isfile(dst_rgb):
            os.remove(dst_rgb)
        os.replace(tmp_rgb, dst_rgb)

    if write_data_once:
        for tmp_name, dst_name in [
            (f"tmp_dpt_{frame_str}.exr", f"{camera_id:03d}_dpt.exr"),
            (f"tmp_nor_{frame_str}.exr", f"{camera_id:03d}_nor.exr"),
            (f"tmp_alb_{frame_str}.png", f"{camera_id:03d}_alb.png"),
            (f"tmp_dif_{frame_str}.png", f"{camera_id:03d}_dif.png"),
        ]:
            tmp = os.path.join(run_dir, tmp_name)
            dst = os.path.join(run_dir, dst_name)
            if os.path.isfile(tmp):
                if os.path.isfile(dst):
                    os.remove(dst)
                os.replace(tmp, dst)

    if write_mask_once:
        tmp_msk = os.path.join(run_dir, f"tmp_msk_{frame_str}.png")
        dst_msk = os.path.join(run_dir, f"{camera_id:03d}_msk.png")
        if os.path.isfile(tmp_msk):
            if os.path.isfile(dst_msk):
                os.remove(dst_msk)
            os.replace(tmp_msk, dst_msk)


# ============================================================
# Main
# ============================================================

def main():
    meta_path = parse_metadata_path_from_argv()
    meta = load_json(meta_path)

    renders = list(meta.get("renders", []))
    if not renders:
        raise RuntimeError("metadata.json has no 'renders' entries.")

    run_dir = os.path.dirname(bpy.path.abspath(meta_path))
    os.makedirs(run_dir, exist_ok=True)

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

    s_people = float(meta.get("S_people", 1.0))
    people_obj.scale = (s_people, s_people, s_people)
    people_obj.rotation_mode = "QUATERNION"

    cameras = meta.get("cameras", [])
    cam_name = cameras[0].get("camera_name", "Camera") if cameras else "Camera"
    cam_obj = ensure_camera_named(cam_name)
    cam_obj.rotation_mode = "QUATERNION"

    apply_intrinsics(cam_obj, cameras[0].get("intrinsics") if cameras else None)

    light_obj = ensure_disk_area_light()
    light_obj.rotation_mode = "QUATERNION"

    target_names = [c.get("target_name") for c in cameras if c.get("target_name")]
    preferred_target_name = target_names[0] if target_names else None
    cam_target = find_cam_target_like_render(cam_obj, preferred_target_name)

    ensure_compositor_nodes(int(meta.get("human_pass_index", 999)), meta.get("rgb_format", "PNG"))

    scene = bpy.context.scene
    frame_max = max(int(r.get("frame", 0)) for r in renders)
    scene.frame_start = 0
    scene.frame_end = max(0, frame_max)

    people_by_key = {
        (int(p["pos_id"]), int(p["yaw_local"])): p
        for p in meta.get("people_samples", [])
        if "pos_id" in p and "yaw_local" in p
    }
    camera_by_id = {
        int(c["camera_id"]): c
        for c in cameras
        if "camera_id" in c
    }

    for render_rec in renders:
        pos_id = int(render_rec.get("pos_id"))
        yaw_local = int(render_rec.get("yaw_local"))
        camera_id = int(render_rec.get("camera_id"))
        light_id = int(render_rec.get("light_id"))
        frame = int(render_rec.get("frame"))

        people_rec = people_by_key.get((pos_id, yaw_local))
        if people_rec is None:
            raise RuntimeError(f"Missing people_samples for pos_id={pos_id}, yaw_local={yaw_local}")

        cam_rec = camera_by_id.get(camera_id)
        if cam_rec is None:
            raise RuntimeError(f"Missing camera record for camera_id={camera_id}")

        people_obj.matrix_world = Matrix(people_rec["matrix_world"])
        cam_obj.matrix_world = Matrix(cam_rec["matrix_world"])
        if "focal_mm" in cam_rec:
            cam_obj.data.lens = float(cam_rec["focal_mm"])

        people_origin = people_obj.matrix_world.translation.copy()
        if cam_target is not None:
            cam_target.location = people_origin

        mode = str(render_rec.get("mode", ""))
        if mode == "no_light":
            light_obj.data.energy = 0.0
        else:
            light_rec = render_rec.get("light")
            if not isinstance(light_rec, dict):
                raise RuntimeError(f"Missing light data for camera_id={camera_id} light_id={light_id}")
            light_obj.matrix_world = Matrix(light_rec["matrix_world"])
            light_obj.data.energy = float(light_rec["power"])
            light_obj.data.size = float(light_rec.get("size_bu", light_rec.get("size_m", 0.0)))
            c = light_rec["color"]
            light_obj.data.color = (float(c[0]), float(c[1]), float(c[2]))

        write_data_once = mode == "no_light"
        write_mask_once = mode == "no_light"

        render_and_rename(run_dir, frame, camera_id, light_id, write_data_once, write_mask_once)

    print(f"[DONE] Output:  {run_dir}", flush=True)
    print(f"[DONE] Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
