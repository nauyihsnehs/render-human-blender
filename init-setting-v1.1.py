import bpy
from math import radians
from contextlib import suppress


def unit_scale(scene=None) -> float:
    scene = scene or bpy.context.scene
    u = getattr(getattr(scene, "unit_settings", None), "scale_length", 1.0) or 1.0
    return u if u > 0 else 1.0


def set_attr(obj, attr, value) -> bool:
    if obj and hasattr(obj, attr):
        with suppress(Exception):
            setattr(obj, attr, value);
            return True
    return False


def set_enum(obj, attr, *values):
    if obj and hasattr(obj, attr):
        for v in values:
            with suppress(Exception):
                setattr(obj, attr, v);
                return v
    return None


def call(fn, *args, **kwargs):
    with suppress(Exception):
        return fn(*args, **kwargs)


def apply(pairs):
    for o, a, v in pairs: set_attr(o, a, v)


def applye(pairs):
    for o, a, vs in pairs: set_enum(o, a, *vs)


def viewport_defaults():
    clip = 1000.0 / unit_scale()
    wm = bpy.context.window_manager
    for win in getattr(wm, "windows", []):
        scr = getattr(win, "screen", None)
        for area in getattr(scr, "areas", []):
            if area.type != "VIEW_3D": continue
            for sp in getattr(area, "spaces", []):
                if sp.type != "VIEW_3D": continue
                set_attr(sp, "clip_end", clip)
                set_attr(getattr(sp, "shading", None), "use_compositor", True)


def clean_compositing_nodes():
    sc = bpy.context.scene
    set_attr(sc, "use_nodes", True)
    nt = getattr(sc, "node_tree", None)
    if not nt: return
    nodes, links = nt.nodes, nt.links
    rl = next((n for n in nodes if n.type == "R_LAYERS"), None) or nodes.new("CompositorNodeRLayers")
    comp = next((n for n in nodes if n.type == "COMPOSITE"), None) or nodes.new("CompositorNodeComposite")
    for n in list(nodes):
        if n.type not in ("R_LAYERS", "COMPOSITE"): call(nodes.remove, n)
    for l in list(links): call(links.remove, l)
    out_img = getattr(rl, "outputs", {}).get("Image")
    in_img = getattr(comp, "inputs", {}).get("Image")
    if out_img and in_img: call(links.new, out_img, in_img)
    set_attr(rl, "location", (-300, 0))
    set_attr(comp, "location", (200, 0))


def cycles_device():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    set_enum(sc.cycles, "device", "GPU", "CPU")
    addon = bpy.context.preferences.addons.get("cycles")
    cp = getattr(addon, "preferences", None) if addon else None
    if not cp or not set_enum(cp, "compute_device_type", "CUDA", "OPTIX"): return
    call(cp.get_devices)
    for d in getattr(cp, "devices", []) or []: set_attr(d, "use", True)


def render_settings():
    sc = bpy.context.scene
    c, r = sc.cycles, sc.render
    apply([
        (c, "use_preview_adaptive_sampling", True), (c, "preview_adaptive_threshold", 0.1),
        (c, "preview_samples", 64), (c, "preview_adaptive_min_samples", 0), (c, "use_preview_denoising", True),
        (c, "use_adaptive_sampling", True), (c, "adaptive_threshold", 0.001),
        (c, "samples", 64),
        # (c, "samples", 256),
        (c, "adaptive_min_samples", 0), (c, "use_denoising", True), (c, "denoising_use_gpu", True),
        (c, "max_bounces", 12), (c, "diffuse_bounces", 4), (c, "glossy_bounces", 4),
        (c, "transmission_bounces", 12), (c, "volume_bounces", 0), (c, "transparent_max_bounces", 8),
        (c, "sample_clamp_direct", 0.0), (c, "sample_clamp_indirect", 5.0),
        (c, "caustics_reflective", True), (c, "caustics_refractive", True),
        (r, "use_simplify", True), (r, "simplify_subdivision_render", 3), (r, "simplify_subdivision", 3),
        (c, "use_auto_tile", False), (c, "use_tiling", False), (r, "use_persistent_data", True),
        (r, "resolution_x", 256), (r, "resolution_y", 256),
        # (r, "resolution_x", 1024), (r, "resolution_y", 1024)
        (r, "resolution_percentage", 100),
        (sc.view_settings, "exposure", 0.0), (sc.view_settings, "gamma", 1.0),
        (sc.view_settings, "use_curve_mapping", False),
    ])
    applye([
        (c, "preview_denoiser", ["OPENIMAGEDENOISE"]),
        (c, "preview_denoising_input_passes", ["RGB_ALBEDO", "ALBEDO"]),
        (c, "preview_denoising_prefilter", ["FAST"]),
        (c, "denoiser", ["OPENIMAGEDENOISE"]),
        (c, "denoising_input_passes", ["RGB_ALBEDO_NORMAL", "ALBEDO_NORMAL", "RGB_ALBEDO"]),
        (c, "denoising_prefilter", ["ACCURATE"]),
        (c, "texture_limit_render", ["2048"]),
        (sc.display_settings, "display_device", ["sRGB"]),
        (sc.view_settings, "view_transform", ["Standard"]),
        (sc.view_settings, "look", ["None", "NONE"]),
        (sc.sequencer_colorspace_settings, "name", ["sRGB"]),
    ])


def view_layer_passes():
    sc = bpy.context.scene
    vl = sc.view_layers[0] if sc.view_layers else None
    if not vl: return
    apply([(vl, "use_pass_z", True), (vl, "use_pass_normal", True)])
    if not set_attr(vl, "use_pass_denoising_data", True):
        set_attr(getattr(vl, "cycles", None), "denoising_store_passes", True)


def world_env_strength_div10():
    sc = bpy.context.scene
    w = getattr(sc, "world", None)
    nt = getattr(w, "node_tree", None) if w and getattr(w, "use_nodes", False) else None
    if not nt or not any(n.type == "TEX_ENVIRONMENT" for n in nt.nodes): return
    for n in nt.nodes:
        if n.type != "BACKGROUND": continue
        s = n.inputs.get("Strength") if hasattr(n, "inputs") else None
        if s and hasattr(s, "default_value"): s.default_value *= 0.25


def sunlight_energy_div10():
    for obj in bpy.data.objects:
        if obj.type == "LIGHT" and getattr(getattr(obj, "data", None), "type", None) == "SUN":
            with suppress(Exception): obj.data.energy *= 0.25


def purge_cameras():
    cams = [o for o in bpy.data.objects if o.type == "CAMERA"]
    empties = set()
    for cam in cams:
        for con in getattr(cam, "constraints", []) or []:
            tgt = getattr(con, "target", None)
            if getattr(tgt, "type", None) == "EMPTY": empties.add(tgt)
        maybe = bpy.data.objects.get(f"{cam.name}.Target")
        if getattr(maybe, "type", None) == "EMPTY": empties.add(maybe)
    for cam in cams: call(bpy.data.objects.remove, cam, do_unlink=True)
    for e in empties: call(bpy.data.objects.remove, e, do_unlink=True)
    for datablock in list(bpy.data.cameras):
        if getattr(datablock, "users", 0) == 0: call(bpy.data.cameras.remove, datablock)


def create_camera(cam_name="Cam001"):
    u = unit_scale()
    purge_cameras()
    sc = bpy.context.scene
    cam_data = bpy.data.cameras.new(cam_name)
    cam_obj = bpy.data.objects.new(cam_name, cam_data)
    sc.collection.objects.link(cam_obj);
    sc.camera = cam_obj
    tgt = bpy.data.objects.new(f"{cam_name}.Target", None)
    sc.collection.objects.link(tgt)
    con = cam_obj.constraints.new("TRACK_TO");
    con.target = tgt
    applye([
        (tgt, "empty_display_type", ["PLAIN_AXES", "ARROWS"]),
        (con, "track_axis", ["TRACK_NEGATIVE_Z", "-Z"]),
        (con, "up_axis", ["UP_Y", "Y"]),
        (con, "target_space", ["WORLD"]), (con, "owner_space", ["WORLD"]),
        (cam_data, "sensor_fit", ["AUTO"]),
    ])
    apply([
        (tgt, "empty_display_size", 1.0),
        (cam_data, "shift_x", 0.0), (cam_data, "shift_y", 0.0),
        (cam_data, "sensor_width", 36.0),
        (cam_data, "clip_start", 0.01 / u), (cam_data, "clip_end", 1000.0 / u),
    ])
    set_attr(cam_data, "angle", radians(60.0))


def surface_chain_nodes(mat):
    nt = getattr(mat, "node_tree", None) if mat and getattr(mat, "use_nodes", False) else None
    if not nt: return []
    nodes = nt.nodes
    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL" and getattr(n, "is_active_output", False)), None) \
          or next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    surf = out.inputs.get("Surface") if out else None
    if not (surf and surf.is_linked): return []
    seen, stack, res = set(), [lk.from_node for lk in surf.links if lk.from_node], []
    while stack:
        n = stack.pop()
        if not n or n in seen: continue
        seen.add(n);
        res.append(n)
        for inp in getattr(n, "inputs", []) or []:
            if inp.is_linked: stack.extend(lk.from_node for lk in inp.links if lk.from_node)
    return res


def fix_brightcontrast_in_surface_chains():
    fixed, seen_mats = 0, set()
    for obj in bpy.data.objects:
        if obj.type != "MESH": continue
        for slot in getattr(obj, "material_slots", []) or []:
            mat = getattr(slot, "material", None)
            if not mat or mat in seen_mats: continue
            for node in surface_chain_nodes(mat):
                if node.type != "BRIGHTCONTRAST" and node.__class__.__name__ != "ShaderNodeBrightContrast": continue
                b = node.inputs.get("Bright") if hasattr(node, "inputs") else None
                c = node.inputs.get("Contrast") if hasattr(node, "inputs") else None
                if not b and hasattr(node, "inputs") and len(node.inputs) > 1: b = node.inputs[1]
                if not c and hasattr(node, "inputs") and len(node.inputs) > 2: c = node.inputs[2]
                if b and hasattr(b, "default_value"): b.default_value = 0.0
                if c and hasattr(c, "default_value"): c.default_value = 0.0
                fixed += 1
            seen_mats.add(mat)
    print(f"[OK] Bright/Contrast reset on {fixed} node(s)." if fixed else "[OK] No Bright/Contrast nodes found in Mesh Surface chains.")


def apply_all():
    viewport_defaults()
    clean_compositing_nodes()
    cycles_device()
    render_settings()
    view_layer_passes()
    world_env_strength_div10()
    sunlight_energy_div10()
    create_camera()
    fix_brightcontrast_in_surface_chains()
    print("\n[DONE] Render/output initialization applied.")


if __name__ == "__main__":
    apply_all()
