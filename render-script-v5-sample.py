import bpy
import os
import json
import math
import random
import sys
import colorsys
import bmesh
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree

# ============================================================
# CONFIG
# ============================================================
DEFAULT_CONFIG = {
    # 人体资产 .blend 路径（Append 导入源）
    # human_object_name 指定要导入的对象名（为空则取第一个对象）
    "human_blend_path": r"E:\render_people\train\00001\00001-3.6.blend",

    # 输出根目录；脚本会在其下创建 run_dir（通常含 scene_id/human_id），并写入 rgb/dpt/nor/alb/msk 与 metadata.json。
    "output_base_path": r"E:\evermotion\Archinteriors-Vol.58-3.6\AI58_002\render",

    # RGB output format: "PNG" (like albedo) or "OPEN_EXR"/"EXR" (like normal/depth)
    "rgb_format": "PNG",
    "scene_id": 0,
    "human_id": 0,

    # 人体对象名（在 human_blend_path 指向的 .blend 内）。
    "human_object_name": "person_render",
    # 人体对象的 pass_index（Object Index pass），Compositor 通过该 ID 生成 msk.png。
    "human_pass_index": 999,
    # 人体整体缩放系数。
    "S_people": 0.01,

    # 采样数量定义：positions 为共享站位数 N；每站位采样若干 yaw；每 yaw 采样若干 camera；
    # 每 camera 包含 1 个 no-light step + N_lights_per_camera 个 lit steps（每个 light 输出一帧 RGB）。
    "N_positions": 10,
    "N_people_yaw_per_pos": 1,
    "N_cameras_per_yaw": 1,
    "N_lights_per_camera": 1,

    # 自动站位采样与贴地：floor_object_name 为地面 mesh 名，用于向下 ray_cast 命中与 XY 边界；
    # ground_epsilon 为贴地后额外抬升量，用于避免脚底穿插/抖动。
    # floor_object_names 若非空列表，会合并为一个名为 floor_object_name 的 mesh
    "floor_object_names": ['AI58_002_Floor', 'AI58_002_Floor_Base'],
    "floor_object_name": "AI58_002_Floor_Full",
    "ground_epsilon": 0.002,
    "require_floor_hit": True,
    "floor_hit_ray_z_top_m": 1000.0,
    # 场景原点固定为世界坐标原点 (0,0,0)；用于“主朝向 main yaw”的计算。

    # yaw 扰动标准差（deg）：最终 yaw = yaw_main + Normal(0, sigma)（并进行角度归一化/截断）。
    "sigma_yaw_deg": 7.5,
    # 相机距离采样：始终使用“焦段驱动距离”逻辑（采样 focal_mm，然后按 ref_distance*(focal/ref_focal) 加扰动）。

    # 焦段区间列表（mm）及其权重；先按权重选区间，再在区间内均匀采样 focal_mm。
    "focal_ranges_mm": [[24.0, 35.0], [35.0, 50.0], [50.0, 85.0]],
    "focal_range_weights": [0.2, 0.5, 0.3],

    # 焦段驱动距离采样的参考点：ref_focal_mm 为参考焦距（mm）；ref_distance_m 为参考距离（m）。
    "ref_focal_mm": 35.0,
    "ref_distance_m": 0.7,  # 0.48~1.7
    # 焦段驱动距离采样的扰动：D *= (1 + U[-r,+r])；裁剪范围由 ref_focal_mm/ref_distance_m 与焦段最小/最大值自动推导。
    "distance_perturb_r": 0.05,

    # 相机/光源点到场景 mesh 的最小安全距离阈值；用于避免贴墙/穿模/不稳定采样。单位m
    "min_clearance_to_scene": 0.05,

    # 光源方向采样（相对相机方向的球坐标角度）：
    # 以概率 p_face 采样 face region：theta ∈ [-theta0,+theta0]，phi ∈ [-phi0,+phi0]；
    # 否则采样全局：theta ∈ [-180,180]，phi ∈ [-90,90]。
    "theta0_deg": 90.0,
    "phi0_deg": 45.0,
    "p_face": 0.9,

    # 光源照度(Illuminance, 照度E)采样范围（脚本按 uniform 采样）。
    # 采样得到：相机-人体距离 D（m）、光源半径 R（m, 由直径/2 得到）、照度 E，计算光源能量：
    #   P = E * pi * (D^2 + R^2)
    # P_min/P_max 已废弃
    "E_min": 2.0,
    "E_max": 16.0,

    # Disk Area Light 的尺寸采样范围（直径），脚本按 uniform 采样。单位m
    "S_min": 0.02,
    "S_max": 0.2,

    # 光色采样模式权重（会自动归一化，总和=1）：
    # white=(1,1,1)；kelvin_common 从 kelvin_common_range 采样；
    # kelvin_other 从 kelvin_other_ranges 采样；rgb 为 HSV 随机（受 rgb_*_range 约束）。
    "w_white": 0.30,
    "w_kelvin_common": 0.40,
    "w_kelvin_other": 0.20,
    "w_rgb": 0.10,

    # 色温采样范围（K）：common 为单一区间；other 为多区间列表（先选区间再采样）。
    "kelvin_common_range": [3500.0, 5600.0],
    "kelvin_other_ranges": [[2700.0, 3500.0], [5600.0, 6500.0]],

    # 色温→RGB 转换的内部 clamp 配置（以脚本实现为准），用于限制极端色温导致的异常颜色/亮度。
    "kelvin_rgb_clamp": [2000.0, 10000.0],

    # RGB（HSV）模式下的采样范围：S 为饱和度范围，V 为明度范围。
    "rgb_saturation_range": [0.6, 1.0],
    "rgb_value_range": [0.7, 1.0],

    # 光源与相机位置的最小距离；用于避免灯贴近镜头造成异常高光/遮挡等问题。
    "min_light_camera_separation": 0.01,

    # 随机种子；控制 positions / yaw / camera / light 的采样可复现。
    "random_seed": 1234,

    "pos_max_attempts": 1000,
    # When a full position (all yaws/cameras/lights) cannot be sampled, choose how to proceed:
    # - "replace": abandon this position and sample a new position to keep exactly N_positions outputs.
    # - "skip": abandon this position and move on (may produce < N_positions outputs).
    "position_failure_policy": "replace",

    # Safety cap for how many total position attempts are allowed when using "replace".
    # Default (if absent) will be N_positions * pos_max_attempts.
    "pos_total_max_attempts": None,

    "cam_max_attempts": 200,
    "light_max_attempts": 200,
}


# ============================================================
# Helpers
# ============================================================
def load_config(default_cfg: dict) -> dict:
    cfg = dict(default_cfg)
    argv = sys.argv
    json_path = None

    if "--" in argv:
        j = argv.index("--") + 1
        if j < len(argv):
            json_path = argv[j]
    elif len(argv) >= 2 and argv[1].lower().endswith(".json"):
        json_path = argv[1]

    if json_path:
        json_path = bpy.path.abspath(json_path)
        if not os.path.isfile(json_path):
            raise RuntimeError(f"Config JSON not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict):
            raise RuntimeError("Config JSON must be a dict/object at top-level.")
        _deep_merge_dict(cfg, overrides)
        cfg["_config_json_path"] = json_path
    else:
        cfg["_config_json_path"] = None

    # Normalize weights
    w_keys = ["w_white", "w_kelvin_common", "w_kelvin_other", "w_rgb"]
    w_sum = sum(float(cfg.get(k)) for k in w_keys)
    if w_sum > 1e-12:
        for k in w_keys:
            cfg[k] = float(cfg.get(k)) / w_sum
    # Normalize focal range weights
    ws = [float(x) for x in cfg.get("focal_range_weights")]
    s = sum(ws)
    if s > 1e-12:
        cfg["focal_range_weights"] = [x / s for x in ws]

    # Backward-compat: accept ref_distance_cm (cm) from older JSON and convert to meters.
    if 'ref_distance_m' not in cfg and 'ref_distance_cm' in cfg:
        try:
            cfg['ref_distance_m'] = float(cfg['ref_distance_cm']) / 100.0
        except Exception:
            pass

    return cfg


def _deep_merge_dict(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def atomic_write_json(path: str, obj: dict, indent: int = 2):
    """Write JSON atomically: write to a temp file then os.replace.
    This avoids corrupt/partial metadata.json if the process crashes mid-write.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(tmp_path, path)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def clamp(x, a, b):
    return max(a, min(b, x))


def deg2rad(d):
    return d * math.pi / 180.0


def depsgraph():
    return bpy.context.evaluated_depsgraph_get()


def scene_unit_scale_length() -> float:
    """Scene Unit scale length (SU): meters per Blender Unit (BU).
    Blender default metric: SU=1.0 => 1 BU = 1 m.
    If a scene is authored in cm-style units, often SU=0.01 => 1 BU = 1 cm.
    """
    us = getattr(bpy.context.scene, 'unit_settings', None)
    su = float(getattr(us, 'scale_length', 1.0) if us else 1.0)
    return su if su > 0 else 1.0


def bu_from_m(val_m: float, SU: float) -> float:
    return float(val_m) / float(SU if SU > 0 else 1.0)


def m_from_bu(val_bu: float, SU: float) -> float:
    return float(val_bu) * float(SU if SU > 0 else 1.0)


def scale_principled_subsurface_radius(objs, SU: float):
    """Scale Principled BSDF 'Subsurface Radius' by 1/SU for meshes whose material assumes 1BU=1m."""
    SU = float(SU if SU > 0 else 1.0)
    if abs(SU - 1.0) < 1e-12:
        return
    for obj in objs:
        if not obj or obj.type != 'MESH':
            continue
        mats = getattr(obj.data, 'materials', None)
        if not mats:
            continue
        for mat in mats:
            if not mat or not getattr(mat, 'use_nodes', False) or not getattr(mat, 'node_tree', None):
                continue
            if mat.get('_PR_ssr_scaled_SU', None) == SU:
                continue
            nt = mat.node_tree
            for node in nt.nodes:
                if node.type != 'BSDF_PRINCIPLED':
                    continue
                # Find the 'Subsurface Radius' socket (name can vary slightly across versions/locales).
                sock = None
                for inp in node.inputs:
                    if 'Subsurface Radius' in inp.name:
                        sock = inp
                        break
                if sock is None or sock.is_linked:
                    continue
                v = sock.default_value
                sock.default_value = (float(v[0]) / SU, float(v[1]) / SU, float(v[2]) / SU)
            mat['_PR_ssr_scaled_SU'] = SU


def kelvin_to_rgb(kelvin: float, clamp_range=(1000.0, 40000.0)):
    t = clamp(kelvin, *clamp_range) / 100.0

    r = 255.0 if t <= 66.0 else 329.698727446 * ((t - 60.0) ** -0.1332047592)

    if t <= 66.0:
        g = 99.4708025861 * math.log(max(1e-8, t)) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)

    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(max(1e-8, t - 10.0)) - 305.0447927307

    return (clamp01(r / 255.0), clamp01(g / 255.0), clamp01(b / 255.0))


def _as_ranges(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
        return [(float(x[0]), float(x[1]))]
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple)):
        return [(float(r[0]), float(r[1])) for r in x if isinstance(r, (list, tuple)) and len(r) == 2]
    return []


def sample_weighted_uniform_in_ranges(rng, ranges, weights=None):
    if not ranges:
        raise RuntimeError("No ranges provided for sampling.")

    if weights and len(weights) == len(ranges):
        u = rng.random()
        c = 0.0
        for (a, b), w in zip(ranges, weights):
            c += w
            if u <= c:
                return rng.uniform(min(a, b), max(a, b))

    a, b = ranges[int(rng.random() * len(ranges))]
    return rng.uniform(min(a, b), max(a, b))


def sample_focal_mm(cfg, rng):
    fr = _as_ranges(cfg.get("focal_ranges_mm"))
    ws = cfg.get("focal_range_weights")
    return float(sample_weighted_uniform_in_ranges(rng, fr, ws))


# def yaw_from_target_dir_for_local_plusY(target_xy: Vector) -> float:
#     t = Vector((target_xy.x, target_xy.y, 0.0))
#     t.normalize()
#     return math.atan2(-t.x, t.y)

def yaw_from_target_dir_for_local_plusX(target_xy):
    t = target_xy.normalized()
    return math.atan2(t.y, t.x)  # makes local +X point to t


def look_at_matrix(camera_pos: Vector, target: Vector, up: Vector = Vector((0, 0, 1))) -> Matrix:
    forward = (target - camera_pos)
    if forward.length < 1e-8:
        forward = Vector((0, 1, 0))
    forward.normalize()

    z_axis = (-forward).normalized()
    x_axis = up.cross(z_axis)
    if x_axis.length < 1e-8:
        x_axis = Vector((0, 1, 0)).cross(z_axis)
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()

    return Matrix((
        (x_axis.x, y_axis.x, z_axis.x, camera_pos.x),
        (x_axis.y, y_axis.y, z_axis.y, camera_pos.y),
        (x_axis.z, y_axis.z, z_axis.z, camera_pos.z),
        (0.0, 0.0, 0.0, 1.0),
    ))


def rotate_about_axis(v, axis, angle_rad):
    v2 = v.copy()
    v2.rotate(Matrix.Rotation(angle_rad, 4, axis.normalized()))
    return v2


def world_bounds_of_objects(objs):
    mins = Vector((1e18, 1e18, 1e18))
    maxs = Vector((-1e18, -1e18, -1e18))

    for o in objs:
        if o.type != 'MESH':
            continue
        for c in o.bound_box:
            wc = o.matrix_world @ Vector(c)
            mins = Vector((min(mins.x, wc.x), min(mins.y, wc.y), min(mins.z, wc.z)))
            maxs = Vector((max(maxs.x, wc.x), max(maxs.y, wc.y), max(maxs.z, wc.z)))

    return mins, maxs


def aabb_overlap(a_min, a_max, b_min, b_max):
    return (a_min.x <= b_max.x and a_max.x >= b_min.x and
            a_min.y <= b_max.y and a_max.y >= b_min.y and
            a_min.z <= b_max.z and a_max.z >= b_min.z)


def bvh_from_object(obj, dg):
    """
    Return BVH in WORLD coordinates (so overlap() works across objects).
    """
    obj_eval = obj.evaluated_get(dg)
    mesh = obj_eval.to_mesh()

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # transform verts into WORLD
        M = obj_eval.matrix_world
        for v in bm.verts:
            v.co = M @ v.co

        bvh = BVHTree.FromBMesh(bm)
        return bvh, obj_eval
    finally:
        bm.free()


def ray_occluded(scene, dg, origin_w: Vector, target_w: Vector, exclude_names=set()):
    direction = target_w - origin_w
    dist = direction.length
    if dist < 1e-6:
        return False
    direction.normalize()

    hit, loc, normal, face_idx, hit_obj, hit_matrix = scene.ray_cast(dg, origin_w, direction, distance=dist - 1e-4)
    return hit and (not hit_obj or hit_obj.name not in exclude_names)


def min_distance_to_scene_meshes(point_w: Vector, scene_meshes, dg):
    best = 1e18
    for o in scene_meshes:
        bvh, obj_eval = bvh_from_object(o, dg)
        nearest = bvh.find_nearest(point_w)  # WORLD now
        if nearest and nearest[0] is not None:
            best = min(best, (nearest[0] - point_w).length)
        obj_eval.to_mesh_clear()
    return best


# ============================================================
# People object management
# ============================================================
def append_people_one_object(blend_path, object_name=""):
    blend_path = bpy.path.abspath(blend_path)

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


def build_scene_mesh_list(exclude_names=set()):
    return [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.name not in exclude_names]


def group_collides_with_scene(people_meshes, scene_meshes):
    dg = depsgraph()
    pmin, pmax = world_bounds_of_objects(people_meshes)

    for s in scene_meshes:
        smin, smax = world_bounds_of_objects([s])
        if not aabb_overlap(pmin, pmax, smin, smax):
            continue

        bvh_s, s_eval = bvh_from_object(s, dg)
        for pm in people_meshes:
            bvh_p, p_eval = bvh_from_object(pm, dg)
            if bvh_p.overlap(bvh_s):
                p_eval.to_mesh_clear()
                s_eval.to_mesh_clear()
                return True
            p_eval.to_mesh_clear()
        s_eval.to_mesh_clear()

    return False


def build_floor_bvh_world(floor_obj_name: str):
    """Build a WORLD-space BVH for the floor object once, for repeated ray casts.

    Returns (bvh, floor_obj_eval). Caller MUST call floor_obj_eval.to_mesh_clear() when done.
    """
    dg = depsgraph()
    floor_obj = bpy.data.objects.get(floor_obj_name)
    if floor_obj is None or floor_obj.type != 'MESH':
        raise RuntimeError(f"Floor object '{floor_obj_name}' missing or not MESH.")

    obj_eval = floor_obj.evaluated_get(dg)
    mesh = obj_eval.to_mesh()

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # transform verts into WORLD
        M = obj_eval.matrix_world
        for v in bm.verts:
            v.co = M @ v.co

        bvh = BVHTree.FromBMesh(bm)
        return bvh, obj_eval
    finally:
        bm.free()


def ground_snap_to_floor_bvh(root_obj, people_meshes, floor_bvh: BVHTree, x, y, epsilon,
                             z_top: float = 1e6, max_dist: float = None):
    """Snap root_obj (and thus the person) to the highest floor surface at (x,y).

    - Ray is cast from (x,y,z_top) straight down.
    - If multiple floors overlap vertically at the same (x,y), the closest hit from above is chosen
      (i.e. the higher floor surface).

    Returns new root_obj.location (Vector) or None if no floor hit.
    """
    if floor_bvh is None:
        raise RuntimeError("floor_bvh is None (did you call build_floor_bvh_world?)")

    if max_dist is None:
        max_dist = float(z_top) * 2.0

    origin = Vector((x, y, float(z_top)))
    direction = Vector((0.0, 0.0, -1.0))
    hit = floor_bvh.ray_cast(origin, direction, float(max_dist))  # WORLD coords

    if hit[0] is None:
        return None

    floor_hit_w = hit[0]

    # First put the root on the hit Z, then raise so the person's lowest point is at hitZ + epsilon.
    root_obj.location = Vector((x, y, float(floor_hit_w.z)))
    bpy.context.view_layer.update()
    mins, _ = world_bounds_of_objects(people_meshes)
    root_obj.location.z += (float(floor_hit_w.z) + float(epsilon)) - float(mins.z)

    return root_obj.location.copy()


# ============================================================
# Camera / Light
# ============================================================
def ensure_camera():
    scene = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None or cam_obj.type != 'CAMERA':
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj
    return cam_obj


def ensure_disk_area_light():
    name = "PeopleDiskLight"
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'LIGHT' and obj.data.type == 'AREA':
        obj.data.shape = 'DISK'
        return obj

    ld = bpy.data.lights.new(name, type='AREA')
    ld.shape = 'DISK'
    obj = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def fov_deg_to_focal_mm(cam_data, fov_deg: float) -> float:
    fov = math.radians(fov_deg)
    sensor_w = float(getattr(cam_data, "sensor_width", 36.0))
    return (sensor_w * 0.5) / max(1e-8, math.tan(fov * 0.5))


def face_forward_world(people_obj) -> Vector:
    v = people_obj.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    v.z = 0.0
    return v.normalized()


def sample_camera(cfg, rng, people_obj, people_origin_w, scene_meshes, exclude_names):
    cam_obj = ensure_camera()
    cam = cam_obj.data
    cam_obj.rotation_mode = 'QUATERNION'

    target_obj = bpy.data.objects.get(f"{cam_obj.name}.Target")

    if hasattr(cam, "lens_unit"):
        cam.lens_unit = 'MILLIMETERS'

    front = face_forward_world(people_obj)

    SU = float(cfg.get("_SU"))
    min_clear_bu = bu_from_m(float(cfg["min_clearance_to_scene"]), SU)

    for st in range(int(cfg["cam_max_attempts"])):
        # Sample focal length (mm), derive camera distance from (ref_focal_mm, ref_distance_m).
        focal_mm = float(sample_focal_mm(cfg, rng))
        f0 = float(cfg.get("ref_focal_mm"))
        d0_m = float(cfg.get("ref_distance_m"))
        d0_bu = bu_from_m(d0_m, SU)
        r = float(cfg.get("distance_perturb_r"))

        fr = _as_ranges(cfg.get("focal_ranges_mm"))
        fmin = min(a for a, _b in fr) if fr else 24.0
        fmax = max(b for _a, b in fr) if fr else 85.0

        dmin_bu = d0_bu * (fmin / max(1e-8, f0))
        dmax_bu = d0_bu * (fmax / max(1e-8, f0))

        D_sampled_bu = d0_bu * (focal_mm / max(1e-8, f0))
        u_perturb = rng.uniform(-r, r)
        D_sampled_bu = clamp(D_sampled_bu * (1.0 + u_perturb), dmin_bu, dmax_bu)

        target_obj.location = people_origin_w

        desired_cam_pos = people_origin_w + front * D_sampled_bu
        desired_cam_pos.z = people_origin_w.z

        cam_obj.location = desired_cam_pos
        bpy.context.view_layer.update()
        cam_pos = cam_obj.matrix_world.translation.copy()
        origin_pos = target_obj.matrix_world.translation.copy()

        scene = bpy.context.scene
        dg = depsgraph()

        if ray_occluded(scene, dg, cam_pos, origin_pos, exclude_names=exclude_names):
            print(f'Failed camera sample attempt {st}: occluded to people.')
            continue
        if min_distance_to_scene_meshes(cam_pos, scene_meshes, dg) < min_clear_bu:
            print(f'Failed camera sample attempt {st}: too close to scene mesh.')
            continue

        cam.lens = float(focal_mm)
        D_actual_bu = (cam_pos - origin_pos).length

        return {
            "distance_m": m_from_bu(float(D_actual_bu), SU),
            "distance_sampled_m": m_from_bu(float(D_sampled_bu), SU),
            "FOV_deg": None,
            "focal_mm": float(focal_mm),
            "focal_mm_sampled": float(focal_mm),
            "distance_perturb_u": float(u_perturb),
            "location": [cam_pos.x, cam_pos.y, cam_pos.z],
            "matrix_world": [list(row) for row in cam_obj.matrix_world],
            "camera_name": cam_obj.name,
            "target_name": (target_obj.name if target_obj is not None else None),
        }

    return None


def sample_light(cfg, rng, people_origin_w, cam_pos, scene_meshes, exclude_names):
    light_obj = ensure_disk_area_light()
    ld = light_obj.data
    light_obj.rotation_mode = 'QUATERNION'
    ld.type = 'AREA'
    ld.shape = 'DISK'

    v0 = (cam_pos - people_origin_w)
    if v0.length < 1e-8:
        v0 = Vector((0, 1, 0))
    v0.normalize()
    D_bu = (cam_pos - people_origin_w).length

    SU = float(cfg.get("_SU"))
    D_m = m_from_bu(float(D_bu), SU)
    min_clear_bu = bu_from_m(float(cfg["min_clearance_to_scene"]), SU)
    min_cam_sep_bu = bu_from_m(float(cfg["min_light_camera_separation"]), SU)

    for st in range(int(cfg["light_max_attempts"])):
        if rng.random() < float(cfg.get("p_face")):
            theta = rng.uniform(-float(cfg["theta0_deg"]), float(cfg["theta0_deg"]))
            phi = rng.uniform(-float(cfg["phi0_deg"]), float(cfg["phi0_deg"]))
        else:
            theta = rng.uniform(-180.0, 180.0)
            phi = rng.uniform(-90.0, 90.0)

        v1 = rotate_about_axis(v0, Vector((0, 0, 1)), deg2rad(theta))
        right = Vector((0, 0, 1)).cross(v1)
        if right.length < 1e-8:
            right = Vector((1, 0, 0))
        right.normalize()
        v2 = rotate_about_axis(v1, right, deg2rad(phi)).normalized()

        light_pos = people_origin_w + v2 * D_bu

        if (light_pos - cam_pos).length < min_cam_sep_bu:
            print(f'Failed light sample attempt {st}: too close to camera.')
            continue

        scene = bpy.context.scene
        dg = depsgraph()

        if ray_occluded(scene, dg, light_pos, people_origin_w, exclude_names=exclude_names):
            print(f'Failed light sample attempt {st}: occluded to people.')
            continue
        if min_distance_to_scene_meshes(light_pos, scene_meshes, dg) < min_clear_bu:
            print(f'Failed light sample attempt {st}: too close to scene mesh.')
            continue

        # Sample light size (diameter, meters) - same as before.
        size_m = rng.uniform(float(cfg["S_min"]), float(cfg["S_max"]))
        size_bu = bu_from_m(size_m, SU)

        # Sample illuminance (E) and convert to light power (P).
        # D: camera-human distance (m); R: light radius (m).
        E_cfg = rng.uniform(float(cfg.get("E_min")), float(cfg.get("E_max")))
        R_m = 0.5 * float(size_m)
        power_cfg = float(E_cfg) * math.pi * (D_m * D_m + R_m * R_m)
        # Compensate for non-meter scene units (legacy logic: power / (SU^2)).
        power = power_cfg / max(1e-12, SU * SU)

        # Light color sampling
        u_mix = rng.random()
        w_white = float(cfg.get("w_white"))
        w_kelvin_common = float(cfg.get("w_kelvin_common"))
        w_kelvin_other = float(cfg.get("w_kelvin_other"))
        w_rgb = float(cfg.get("w_rgb"))

        kelvin = None

        if u_mix < w_white:
            color = (1.0, 1.0, 1.0)
            color_mode = "white"
        elif u_mix < (w_white + w_rgb):
            s_lo, s_hi = cfg.get("rgb_saturation_range")
            v_lo, v_hi = cfg.get("rgb_value_range")
            h = rng.random()
            s = rng.uniform(float(s_lo), float(s_hi))
            v = rng.uniform(float(v_lo), float(v_hi))
            color = colorsys.hsv_to_rgb(h, clamp01(s), clamp01(v))
            color_mode = "rgb"
        else:
            clamp_k = cfg.get("kelvin_rgb_clamp")
            common = _as_ranges(cfg.get("kelvin_common_range"))
            other = _as_ranges(cfg.get("kelvin_other_ranges"))

            k_sum = max(1e-12, w_kelvin_common + w_kelvin_other)
            p_common = w_kelvin_common / k_sum

            if rng.random() < p_common:
                kelvin = float(sample_weighted_uniform_in_ranges(rng, common))
                color_mode = "kelvin_common"
            else:
                kelvin = float(sample_weighted_uniform_in_ranges(rng, other))
                color_mode = "kelvin_other"

            color = kelvin_to_rgb(kelvin, clamp_range=clamp_k)

        light_obj.matrix_world = look_at_matrix(light_pos, people_origin_w)
        ld.energy = float(power)
        ld.size = float(size_bu)
        ld.color = (float(color[0]), float(color[1]), float(color[2]))

        return {
            "theta_deg": float(theta),
            "phi_deg": float(phi),
            "power": float(power),
            "power_cfg": float(power_cfg),
            "illuminance_E": float(E_cfg),
            "distance_D_m": float(D_m),
            "radius_R_m": float(R_m),
            "size_m": float(size_m),
            "size_bu": float(size_bu),
            "color": [float(color[0]), float(color[1]), float(color[2])],
            "color_mode": color_mode,
            "kelvin": (float(kelvin) if kelvin is not None else None),
            "location": [light_pos.x, light_pos.y, light_pos.z],
            "matrix_world": [list(row) for row in light_obj.matrix_world],
        }

    return None


# ============================================================
# Compositor
# ============================================================
def ensure_compositor_nodes(human_pass_index: int, rgb_format: str = "PNG"):
    scene = bpy.context.scene
    scene.use_nodes = True
    nt = scene.node_tree
    nodes = nt.nodes
    links = nt.links

    scene.view_layers[0].use_pass_object_index = True

    # Diffuse Color pass
    if hasattr(scene.view_layers[0], "use_pass_diffuse_color"):
        scene.view_layers[0].use_pass_diffuse_color = True

    # Remove old nodes
    for n in list(nodes):
        if (n.type == 'OUTPUT_FILE' or n.type == 'ID_MASK') and n.name.startswith("PR_"):
            nodes.remove(n)

    rl = nodes.get("Render Layers")
    if rl is None:
        rl = nodes.new("CompositorNodeRLayers")

    # RGB (PNG or EXR)
    out_rgb = nodes.new("CompositorNodeOutputFile")
    out_rgb.name = "PR_RGB"
    out_rgb.base_path = ""
    fmt = str(rgb_format or "PNG").upper()
    if fmt in ("EXR", "OPEN_EXR", "OPENEXR"):
        out_rgb.format.file_format = 'OPEN_EXR'
        out_rgb.format.color_mode = 'RGB'
        out_rgb.format.color_depth = '16'
        if hasattr(out_rgb.format, "exr_codec"):
            out_rgb.format.exr_codec = 'ZIP'
    else:
        out_rgb.format.file_format = 'PNG'
        out_rgb.format.color_mode = 'RGB'
        out_rgb.format.color_depth = '8'
    out_rgb.file_slots.clear()
    out_rgb.file_slots.new("Image")
    out_rgb.file_slots[-1].path = "tmp_rgb_####"

    # Depth/Normal (EXR)
    out_dat = nodes.new("CompositorNodeOutputFile")
    out_dat.name = "PR_DATA"
    out_dat.base_path = ""
    out_dat.format.file_format = 'OPEN_EXR'
    out_dat.format.color_mode = 'RGB'
    out_dat.format.color_depth = '32'
    if hasattr(out_dat.format, "exr_codec"):
        out_dat.format.exr_codec = 'ZIP'
    out_dat.file_slots.clear()
    out_dat.file_slots.new("Depth")
    out_dat.file_slots[-1].path = "tmp_dpt_####"
    out_dat.file_slots.new("Normal")
    out_dat.file_slots[-1].path = "tmp_nor_####"

    # Albedo (PNG)
    out_alb = nodes.new("CompositorNodeOutputFile")
    out_alb.name = "PR_ALB"
    out_alb.base_path = ""
    out_alb.format.file_format = 'PNG'
    out_alb.format.color_mode = 'RGB'
    out_alb.format.color_depth = '8'
    out_alb.file_slots.clear()
    out_alb.file_slots.new("Denoising Albedo")
    out_alb.file_slots[-1].path = "tmp_alb_####"

    # Diffuse Color (PNG)
    out_dif = nodes.new("CompositorNodeOutputFile")
    out_dif.name = "PR_DIF"
    out_dif.base_path = ""
    out_dif.format.file_format = 'PNG'
    out_dif.format.color_mode = 'RGB'
    out_dif.format.color_depth = '8'
    out_dif.file_slots.clear()
    out_dif.file_slots.new("Diffuse Color")
    out_dif.file_slots[-1].path = "tmp_dif_####"

    # Mask
    idm = nodes.new("CompositorNodeIDMask")
    idm.name = "PR_IDMASK"
    idm.index = int(human_pass_index)
    if hasattr(idm, "use_antialiasing"):
        idm.use_antialiasing = True

    out_msk = nodes.new("CompositorNodeOutputFile")
    out_msk.name = "PR_MSK"
    out_msk.base_path = ""
    out_msk.format.file_format = 'PNG'
    out_msk.format.color_mode = 'BW'
    out_msk.format.color_depth = '8'
    out_msk.file_slots.clear()
    out_msk.file_slots.new("Mask")
    out_msk.file_slots[-1].path = "tmp_msk_####"

    # Links
    def safe_link(out_name, node, in_name):
        s = rl.outputs.get(out_name)
        i = node.inputs.get(in_name)
        if s and i:
            links.new(s, i)

    safe_link("Image", out_rgb, "Image")
    safe_link("Depth", out_dat, "Depth")
    safe_link("Normal", out_dat, "Normal")
    safe_link("Denoising Albedo", out_alb, "Denoising Albedo")
    # Diffuse Color output name differs by Blender version
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

    # FORCE refresh (important for persistent data)
    scene.node_tree.update_tag()
    bpy.context.view_layer.update()

    scene.frame_set(frame_idx)
    bpy.context.view_layer.update()

    bpy.ops.render.render(write_still=False, use_viewport=False)

    frame_str = f"{frame_idx:04d}"

    # RGB (PNG or EXR)
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
# Placement
# ============================================================


def join_floor_meshes_world(floor_names, joined_name="PR_FLOOR_JOINED"):
    """Join multiple floor meshes into a single WORLD-space mesh object.

    - Preserves materials (best-effort) so the joined floor can be rendered.
    - Geometry is baked into world coordinates and the joined object's matrix_world is Identity.
    """
    dg = depsgraph()

    bm_all = bmesh.new()
    mats = []  # global material list
    mat_index = {}  # material pointer -> global index

    def _mat_global_idx(mat):
        if mat is None:
            return 0
        key = mat.as_pointer()
        if key in mat_index:
            return mat_index[key]
        mats.append(mat)
        mat_index[key] = len(mats) - 1
        return mat_index[key]

    for nm in floor_names:
        obj = bpy.data.objects.get(nm)
        if obj is None or obj.type != 'MESH':
            continue

        obj_eval = obj.evaluated_get(dg)
        mesh = obj_eval.to_mesh()

        bm = bmesh.new()
        try:
            bm.from_mesh(mesh)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            # WORLD transform baked into verts
            M = obj_eval.matrix_world
            vmap = {}
            for v in bm.verts:
                nv = bm_all.verts.new(M @ v.co)
                vmap[v.index] = nv
            bm_all.verts.ensure_lookup_table()

            # per-object material mapping: local index -> global index
            local_mats = list(getattr(obj, "data", None).materials) if getattr(obj, "data", None) else []
            local_to_global = {}
            for li, mat in enumerate(local_mats):
                local_to_global[li] = _mat_global_idx(mat)

            for f in bm.faces:
                try:
                    nf = bm_all.faces.new([vmap[v.index] for v in f.verts])
                except ValueError:
                    # face exists; ignore
                    continue
                # material index (fallback to 0 if missing)
                li = int(getattr(f, "material_index", 0))
                nf.material_index = int(local_to_global.get(li, 0))
        finally:
            bm.free()
            obj_eval.to_mesh_clear()

    bm_all.verts.ensure_lookup_table()
    bm_all.faces.ensure_lookup_table()

    # Replace old joined object if exists
    old = bpy.data.objects.get(joined_name)
    if old is not None:
        if old.data:
            bpy.data.meshes.remove(old.data, do_unlink=True)
        bpy.data.objects.remove(old, do_unlink=True)

    mesh_new = bpy.data.meshes.new(joined_name + "_MESH")
    bm_all.to_mesh(mesh_new)
    bm_all.free()

    # Attach materials (slot order matches global indices used above)
    if mats:
        for mat in mats:
            if mat is not None:
                mesh_new.materials.append(mat)

    obj_new = bpy.data.objects.new(joined_name, mesh_new)
    bpy.context.scene.collection.objects.link(obj_new)
    obj_new.matrix_world = Matrix.Identity(4)
    return obj_new


def resolve_floor_object(cfg):
    """Support floor list: cfg['floor_object_names'] (preferred) or cfg['floor_object_name'] (fallback).

    If floor_object_names is provided, a new joined floor mesh is created in WORLD space (for sampling/raycasting),
    and the source floor objects are disabled for rendering (hide_render=True) to avoid double geometry.
    """
    names = cfg.get("floor_object_names")
    if isinstance(names, (list, tuple)) and len(names) > 0:
        src_names = [str(x) for x in names]
        joined = join_floor_meshes_world(src_names, joined_name=str(cfg.get('floor_object_name')))
        # Disable original floor objects in renders to avoid duplicates
        for nm in src_names:
            o = bpy.data.objects.get(nm)
            if o is not None:
                o.hide_render = True
        # Ensure joined floor renders (it carries the geometry now)
        joined.hide_render = False
        # cfg["floor_object_name"] = joined.name
        return joined.name
    return cfg.get("floor_object_name")


def floor_xy_bounds_world(floor_obj_name: str):
    floor = bpy.data.objects.get(floor_obj_name)
    if floor is None or floor.type != 'MESH':
        raise RuntimeError(f"Floor object '{floor_obj_name}' missing or not MESH.")

    xs = [wc.x for c in floor.bound_box for wc in [floor.matrix_world @ Vector(c)]]
    ys = [wc.y for c in floor.bound_box for wc in [floor.matrix_world @ Vector(c)]]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if (x_max - x_min) < 1e-6 or (y_max - y_min) < 1e-6:
        raise RuntimeError(f"Floor bbox too small: x[{x_min},{x_max}] y[{y_min},{y_max}]")

    return x_min, x_max, y_min, y_max


def xy_hits_floor(floor_obj_name: str, x: float, y: float, z_top: float = 1e6):
    dg = depsgraph()
    floor_obj = bpy.data.objects.get(floor_obj_name)
    if floor_obj is None or floor_obj.type != 'MESH':
        return False

    bvh, floor_eval = bvh_from_object(floor_obj, dg)  # WORLD BVH
    origin = Vector((x, y, z_top))
    direction = Vector((0, 0, -1))
    hit = bvh.ray_cast(origin, direction, z_top * 2.0)
    floor_eval.to_mesh_clear()
    return hit[0] is not None


def sample_shared_positions(cfg, rng, people_obj, people_meshes, scene_meshes,
                            floor_bvh: BVHTree, x_min, x_max, y_min, y_max,
                            require_floor_hit: bool, z_top_bu: float, ground_epsilon_bu: float):
    """Sample shared positions across multi-floor layouts.

    Loop:
      sample (x,y) -> snap to floor to get z (chooses highest floor if overlapping) -> collision -> accept/continue
    """
    positions = []

    init_loc = people_obj.location.copy()
    init_rot_mode = people_obj.rotation_mode
    init_rot = people_obj.rotation_euler.copy()
    init_scale = people_obj.scale.copy()

    try:
        attempts = 0
        max_attempts = int(cfg.get("pos_max_attempts", 1000))
        while len(positions) < int(cfg["N_positions"]) and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(float(x_min), float(x_max))
            y = rng.uniform(float(y_min), float(y_max))

            # Snap to (potentially different) floor height at this XY.
            loc = ground_snap_to_floor_bvh(
                people_obj, people_meshes, floor_bvh,
                x, y, ground_epsilon_bu,
                z_top=float(z_top_bu),
                max_dist=float(z_top_bu) * 2.0
            )
            bpy.context.view_layer.update()

            if loc is None:
                if require_floor_hit:
                    print(f'Failed shared position attempt {attempts}: no floor hit at x={x}, y={y}.')
                    continue
                else:
                    # Keep current Z if floor hit isn't required (legacy behavior).
                    people_obj.location.x = x
                    people_obj.location.y = y
                    bpy.context.view_layer.update()

            if group_collides_with_scene(people_meshes, scene_meshes):
                print(f'Failed shared position attempt {attempts}: collision at x={x}, y={y}, z={people_obj.location.z}.')
                continue

            positions.append({"x": float(people_obj.location.x),
                              "y": float(people_obj.location.y),
                              "z": float(people_obj.location.z)})

        if len(positions) < int(cfg["N_positions"]):
            raise RuntimeError(f"Could not sample N_positions={cfg['N_positions']} placements (got {len(positions)}).")

        return positions

    finally:
        people_obj.location = init_loc
        people_obj.scale = init_scale
        people_obj.rotation_mode = init_rot_mode
        people_obj.rotation_euler = init_rot
        bpy.context.view_layer.update()


class _PositionRetry(Exception):
    """Internal control-flow exception to abandon a single position and resample."""
    pass


def sample_one_position(cfg, rng, people_obj, people_meshes, scene_meshes,
                        floor_bvh: BVHTree, x_min, x_max, y_min, y_max,
                        require_floor_hit: bool, z_top_bu: float, ground_epsilon_bu: float):
    """Sample a single valid (x,y,z) placement (multi-floor aware). Returns dict or None."""
    max_attempts = int(cfg.get("pos_max_attempts", 1000))
    for _ in range(max_attempts):
        x = rng.uniform(float(x_min), float(x_max))
        y = rng.uniform(float(y_min), float(y_max))

        loc = ground_snap_to_floor_bvh(
            people_obj, people_meshes, floor_bvh,
            x, y, ground_epsilon_bu,
            z_top=float(z_top_bu),
            max_dist=float(z_top_bu) * 2.0
        )
        bpy.context.view_layer.update()

        if loc is None:
            if require_floor_hit:
                print(f'Failed one position attempt: no floor hit at x={x}, y={y}.')
                continue
            else:
                people_obj.location.x = x
                people_obj.location.y = y
                bpy.context.view_layer.update()

        if group_collides_with_scene(people_meshes, scene_meshes):
            print(f'Failed one position attempt: collision at x={x}, y={y}, z={people_obj.location.z}.')
            continue

        return {"x": float(people_obj.location.x),
                "y": float(people_obj.location.y),
                "z": float(people_obj.location.z)}

    return None


def set_people_yaw_keep_base_xy(cfg, rng, people_obj, base_rot_euler, scene_origin_w):
    P = people_obj.matrix_world.translation.copy()
    d = scene_origin_w - P
    d.z = 0.0
    d.normalize()

    yaw_main = yaw_from_target_dir_for_local_plusX(-d)
    delta = clamp(rng.gauss(0.0, cfg["sigma_yaw_deg"]), -15.0, 15.0)
    yaw = yaw_main + deg2rad(delta)

    people_obj.rotation_mode = 'XYZ'
    people_obj.rotation_euler = (base_rot_euler.x, base_rot_euler.y, base_rot_euler.z + yaw)
    return float(yaw), float(delta)


# ============================================================
# Main
# ============================================================
def main(cfg):
    human_blend = bpy.path.abspath(cfg["human_blend_path"])
    out_base = bpy.path.abspath(cfg["output_base_path"])

    if not os.path.isfile(human_blend):
        raise RuntimeError("human_blend_path not found.")
    os.makedirs(out_base, exist_ok=True)

    n_pos = int(cfg["N_positions"])
    n_yaw = int(cfg.get("N_people_yaw_per_pos"))
    n_cam = int(cfg.get("N_cameras_per_yaw"))
    n_lgt = int(cfg.get("N_lights_per_camera"))

    if n_cam <= 0 or n_lgt <= 0 or n_pos <= 0 or n_yaw <= 0:
        raise RuntimeError("N_positions / N_people_yaw_per_pos / N_cameras_per_yaw / N_lights_per_camera must be > 0.")

    scene_origin = Vector((0.0, 0.0, 0.0))

    # Standardized units: config lengths are in meters.
    # Scene Unit scale length (SU) tells how many meters per Blender Unit (BU).
    SU = scene_unit_scale_length()
    cfg["_SU"] = SU

    run_dir = os.path.join(out_base, f"{int(cfg['scene_id']):03d}_{int(cfg['human_id']):03d}")
    os.makedirs(run_dir, exist_ok=True)

    people_obj = append_people_one_object(human_blend, cfg.get("human_object_name"))
    s = float(cfg["S_people"]) / SU
    people_obj.scale = (s, s, s)
    people_obj.pass_index = int(cfg.get("human_pass_index"))

    people_meshes = [people_obj] if people_obj.type == 'MESH' else []
    if not people_meshes:
        raise RuntimeError(f"Imported people object is not a MESH: {people_obj.name} ({people_obj.type})")

    scale_principled_subsurface_radius(people_meshes, SU)
    base_rot = people_obj.rotation_euler.copy()

    # Resolve floor (support multi-mesh floors via floor_object_names).
    resolve_floor_object(cfg)

    # Build floor BVH once (WORLD space) so we can repeatedly snap to multi-floor heights.
    floor_bvh, floor_eval = build_floor_bvh_world(cfg["floor_object_name"])

    # Initial snap at current XY (mainly to ensure floor BVH is valid / not empty).
    # x0, y0 = float(people_obj.location.x), float(people_obj.location.y)
    # loc0 = ground_snap_to_floor_bvh(
    #     people_obj, people_meshes, floor_bvh,
    #     x0, y0, bu_from_m(float(cfg["ground_epsilon"]), SU),
    #     z_top=bu_from_m(float(cfg.get("floor_hit_ray_z_top_m")), SU)
    # )
    # if loc0 is None:
    #     raise RuntimeError(
    #         f"Initial floor snap failed at x={x0:.3f}, y={y0:.3f} for floor '{cfg['floor_object_name']}'"
    #     )

    # Exclude floor(s) from collision BVHs (standing on floors should not count as collision).
    # If a floor list was provided, also exclude all source floor objects.
    exclude = {people_obj.name, "Camera", "PeopleDiskLight", cfg["floor_object_name"]}
    floor_src = cfg.get("floor_object_names")
    if isinstance(floor_src, (list, tuple)) and len(floor_src) > 0:
        exclude |= {str(x) for x in floor_src}
    scene_meshes = build_scene_mesh_list(exclude_names=exclude)

    # Bounds for XY sampling (covers all floors if the floor is a joined multi-floor mesh).
    x_min, x_max, y_min, y_max = floor_xy_bounds_world(cfg["floor_object_name"])
    require_floor_hit = bool(cfg.get("require_floor_hit"))
    z_top_bu = bu_from_m(float(cfg.get("floor_hit_ray_z_top_m")), SU)
    ground_epsilon_bu = bu_from_m(float(cfg.get("ground_epsilon")), SU)

    pos_rng = random.Random(int(cfg["random_seed"]))
    positions_candidates = sample_shared_positions(
        cfg, pos_rng, people_obj, people_meshes, scene_meshes,
        floor_bvh, x_min, x_max, y_min, y_max,
        require_floor_hit, z_top_bu, ground_epsilon_bu
    )

    # Normalize RGB format + extension for metadata naming
    _rgb_fmt = str(cfg.get("rgb_format", "PNG") or "PNG").upper()
    rgb_ext = ".exr" if _rgb_fmt in ("EXR", "OPEN_EXR", "OPENEXR") else ".png"

    cam_obj = ensure_camera()
    cam_obj.rotation_mode = 'QUATERNION'
    light_obj = ensure_disk_area_light()
    light_obj.rotation_mode = 'QUATERNION'

    cam_target = bpy.data.objects.get(f"{cam_obj.name}.Target")

    for b in (people_obj, cam_obj, light_obj, cam_obj.data, light_obj.data):
        b.animation_data_clear()
    if cam_target is not None:
        cam_target.animation_data_clear()

    scene = bpy.context.scene

    steps_per_camera = 1 + n_lgt
    frames_per_yaw = n_cam * steps_per_camera
    frames_per_pos = n_yaw * frames_per_yaw
    total_frames = n_pos * frames_per_pos
    scene.frame_start = 0
    scene.frame_end = max(0, total_frames - 1)

    master_rng = random.Random(int(cfg["random_seed"]))

    def new_rng():
        return random.Random(master_rng.getrandbits(64))

    meta = {
        "scene_id": int(cfg["scene_id"]),
        "human_id": int(cfg["human_id"]),
        "human_blend_path": human_blend,
        "human_object_name": cfg.get("human_object_name"),
        "config_json_path": cfg.get("_config_json_path"),
        "S_people": s,
        "rgb_format": _rgb_fmt,
        "scene_origin": [scene_origin.x, scene_origin.y, scene_origin.z],
        "shared_positions": [],
        "counts": {
            "N_positions": n_pos,
            "N_people_yaw_per_pos": n_yaw,
            "N_cameras_per_yaw": n_cam,
            "N_lights_per_camera": n_lgt,
            "no_light_per_camera": True,
            "mask_per_camera": True,
        },
        "frame_indexing": {
            "steps_per_camera": steps_per_camera,
            "frames_per_yaw": frames_per_yaw,
            "frames_per_pos": frames_per_pos,
        },
        "render_resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
        "scene_unit_scale_length": float(SU),
        "human_pass_index": int(cfg.get("human_pass_index")),
        "people_samples": [],
        "cameras": [],
        "renders": [],
    }

    meta_path = os.path.join(run_dir, "metadata.json")
    # Initial write so you can inspect counts/positions even if sampling fails later.
    atomic_write_json(meta_path, meta, indent=2)

    # ========================================================
    # Position rendering with robust resampling
    # If a position cannot produce ALL required yaws/cameras/lights, we abandon it
    # and sample a replacement (default) instead of aborting the whole run.
    # ========================================================

    failure_policy = str(cfg.get("position_failure_policy", "replace") or "replace").lower()

    # Recompute bounds/params here as well (keeps behavior robust if future edits move code around).
    x_min, x_max, y_min, y_max = floor_xy_bounds_world(cfg["floor_object_name"])
    require_floor_hit = bool(cfg.get("require_floor_hit"))
    z_top_bu = bu_from_m(float(cfg.get("floor_hit_ray_z_top_m")), SU)
    ground_epsilon_bu = bu_from_m(float(cfg.get("ground_epsilon")), SU)

    # Total attempts cap when replacing failed positions
    pos_total_max_attempts = cfg.get("pos_total_max_attempts")
    if pos_total_max_attempts is None:
        pos_total_max_attempts = int(n_pos) * int(cfg.get("pos_max_attempts", 1000))
    else:
        pos_total_max_attempts = int(pos_total_max_attempts)

    candidate_i = 0
    total_pos_attempts = 0

    while len(meta["shared_positions"]) < n_pos:
        pos_id = len(meta["shared_positions"])
        total_pos_attempts += 1
        if failure_policy == "replace" and total_pos_attempts > pos_total_max_attempts:
            raise RuntimeError(
                f"Could not complete {n_pos} valid positions after {total_pos_attempts} attempts. "
                f"Consider increasing pos_total_max_attempts/pos_max_attempts or loosening constraints."
            )

        # Pick from the initial candidate pool first; afterwards, sample replacements on demand.
        if candidate_i < len(positions_candidates):
            p = positions_candidates[candidate_i]
            candidate_i += 1
        else:
            rng_pos = new_rng()
            p = sample_one_position(cfg, rng_pos, people_obj, people_meshes, scene_meshes,
                                    floor_bvh, x_min, x_max, y_min, y_max,
                                    require_floor_hit, z_top_bu, ground_epsilon_bu)
            if p is None:
                if failure_policy == "skip":
                    break
                continue

        # Snapshots for rollback if we abandon this position mid-way
        snap_shared = len(meta["shared_positions"])
        snap_people = len(meta["people_samples"])
        snap_cams = len(meta["cameras"])
        snap_renders = len(meta["renders"])

        def _rollback_abandoned_position():
            # Remove metadata appended during this position attempt
            meta["shared_positions"][:] = meta["shared_positions"][:snap_shared]
            meta["people_samples"][:] = meta["people_samples"][:snap_people]
            meta["cameras"][:] = meta["cameras"][:snap_cams]
            meta["renders"][:] = meta["renders"][:snap_renders]

            # Flush after rollback so metadata matches on-disk outputs
            atomic_write_json(meta_path, meta, indent=2)

        try:
            # Record the position immediately so progress is visible even if a crash happens later.
            meta["shared_positions"].append(p)
            atomic_write_json(meta_path, meta, indent=2)

            print(f"[POS] {pos_id + 1}/{n_pos} placing", flush=True)

            people_obj.location = Vector((p["x"], p["y"], p["z"]))
            bpy.context.view_layer.update()
            exclude_now = {people_obj.name, "Camera", "PeopleDiskLight"}

            for yaw_local in range(n_yaw):
                print(f"  [YAW] {yaw_local + 1}/{n_yaw} sampling", flush=True)

                rng_yaw = new_rng()
                yaw_rad, delta_deg = set_people_yaw_keep_base_xy(cfg, rng_yaw, people_obj, base_rot, scene_origin)
                bpy.context.view_layer.update()

                people_origin = people_obj.matrix_world.translation.copy()

                meta["people_samples"].append({
                    "pos_id": pos_id,
                    "yaw_local": yaw_local,
                    "location": [people_obj.location.x, people_obj.location.y, people_obj.location.z],
                    "yaw_rad": yaw_rad,
                    "delta_yaw_deg": delta_deg,
                    "matrix_world": [list(r) for r in people_obj.matrix_world],
                })

                for cam_local in range(n_cam):
                    print(f"    [CAM] {cam_local + 1}/{n_cam} sampling", flush=True)

                    rng_cam = new_rng()
                    cam_rec = sample_camera(cfg, rng_cam, people_obj, people_origin, scene_meshes, exclude_now)
                    if cam_rec is None:
                        raise _PositionRetry(f"camera sampling failed at pos_id={pos_id}, yaw_local={yaw_local}, cam_local={cam_local}")

                    camera_id = (pos_id * n_yaw + yaw_local) * n_cam + cam_local

                    meta["cameras"].append({
                        "pos_id": pos_id,
                        "yaw_local": yaw_local,
                        "cam_local": cam_local,
                        "camera_id": camera_id,
                        **cam_rec,
                        "intrinsics": {
                            "lens_unit": getattr(cam_obj.data, "lens_unit", None),
                            "lens_mm": getattr(cam_obj.data, "lens", None),
                            "sensor_width": getattr(cam_obj.data, "sensor_width", None),
                            "sensor_fit": getattr(cam_obj.data, "sensor_fit", None),
                            "shift_x": getattr(cam_obj.data, "shift_x", None),
                            "shift_y": getattr(cam_obj.data, "shift_y", None),
                        },
                        "per_camera_outputs": {
                            "no_light_rgb": f"{camera_id:03d}_999_rgb{rgb_ext}",
                            "depth": f"{camera_id:03d}_dpt.exr",
                            "normal": f"{camera_id:03d}_nor.exr",
                            "albedo": f"{camera_id:03d}_alb.png",
                            "diffuse": f"{camera_id:03d}_dif.png",
                            "mask": f"{camera_id:03d}_msk.png",
                        }
                    })

                    cam_mw = Matrix(cam_rec["matrix_world"])
                    cam_obj.matrix_world = cam_mw
                    if "focal_mm" in cam_rec:
                        cam_obj.data.lens = float(cam_rec["focal_mm"])

                    cam_pos = Vector(cam_rec["location"])

                    # No-light render (metadata only)
                    step_local = 0
                    frame = pos_id * frames_per_pos + yaw_local * frames_per_yaw + cam_local * steps_per_camera + step_local
                    meta["renders"].append({
                        "frame": frame,
                        "pos_id": pos_id,
                        "yaw_local": yaw_local,
                        "cam_local": cam_local,
                        "camera_id": camera_id,
                        "light_id": 999,
                        "mode": "no_light",
                        "files": {
                            "rgb": f"{camera_id:03d}_999_rgb{rgb_ext}",
                            "depth": f"{camera_id:03d}_dpt.exr",
                            "normal": f"{camera_id:03d}_nor.exr",
                            "albedo": f"{camera_id:03d}_alb.png",
                            "diffuse": f"{camera_id:03d}_dif.png",
                            "mask": f"{camera_id:03d}_msk.png",
                        }
                    })

                    # Flush metadata after each sampled frame (robust to crashes)
                    atomic_write_json(meta_path, meta, indent=2)

                    # Lit renders
                    for light_local in range(n_lgt):
                        step_local = 1 + light_local
                        frame = pos_id * frames_per_pos + yaw_local * frames_per_yaw + cam_local * steps_per_camera + step_local
                        light_id = light_local

                        print(f"      [LGT] {light_local + 1}/{n_lgt} (frame {frame})", flush=True)

                        rng_lgt = new_rng()
                        light_rec = sample_light(cfg, rng_lgt, people_origin, cam_pos, scene_meshes, exclude_now)
                        if light_rec is None:
                            raise _PositionRetry(f"light sampling failed at camera_id={camera_id}, light_local={light_local}")

                        meta["renders"].append({
                            "frame": frame,
                            "pos_id": pos_id,
                            "yaw_local": yaw_local,
                            "cam_local": cam_local,
                            "camera_id": camera_id,
                            "light_id": light_id,
                            "mode": "lit",
                            "light": light_rec,
                            "files": {"rgb": f"{camera_id:03d}_{light_id:03d}_rgb{rgb_ext}"}
                        })

                        # Flush metadata after each sampled frame (robust to crashes)
                        atomic_write_json(meta_path, meta, indent=2)



        except _PositionRetry as e:
            print(f"[WARN] Abandoning position pos_id={pos_id} ({e}).", flush=True)
            _rollback_abandoned_position()
            if failure_policy == "skip":
                # Stop early; metadata will reflect fewer than N_positions.
                break
            continue

    # Final flush
    meta_path = os.path.join(run_dir, "metadata.json")
    atomic_write_json(meta_path, meta, indent=2)

    # Release evaluated floor mesh created by build_floor_bvh_world()
    try:
        floor_eval.to_mesh_clear()
    except Exception:
        pass

    print(f"[DONE] Output:  {run_dir}", flush=True)
    print(f"[DONE] Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    cfg = load_config(DEFAULT_CONFIG)
    main(cfg)
