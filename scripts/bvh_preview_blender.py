"""Render a BVH as a stick figure with Blender (headless), for visual QC of a capture.

  blender -b --factory-startup -P bvh_preview_blender.py
Env: BVH_PATH, OUT_DIR, PREVIEW_END (optional last frame), RES (e.g. 960x540)
"""
import os

import bpy
from mathutils import Matrix, Vector

BVH = os.environ.get("BVH_PATH", r"C:\Users\princ\Downloads\taichi_enh\mocap\taichi60.bvh")
OUT = os.environ.get("OUT_DIR", r"C:\Users\princ\Downloads\taichi_enh\preview")
END = int(os.environ.get("PREVIEW_END", "0"))
RES = os.environ.get("RES", "960x540")
FPS = int(round(float(os.environ.get("FPS", "15"))))

# ---- scene ---------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_anim.bvh(filepath=BVH, global_scale=0.01, axis_forward="-Z", axis_up="Y")
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")

scene = bpy.context.scene
scene.render.fps = FPS
last = END if END else int(arm.animation_data.action.frame_range[1])
scene.frame_start, scene.frame_end = 1, last

world = bpy.data.worlds.new("W")
world.use_nodes = False
world.color = (0.05, 0.06, 0.08)
scene.world = world

scene.render.engine = "BLENDER_WORKBENCH"
sh = scene.display.shading
sh.light, sh.color_type = "STUDIO", "OBJECT"
sh.show_shadows = True
sh.show_cavity = True
sh.cavity_type = "BOTH"
sh.background_type = "WORLD"

w, h = (int(v) for v in RES.split("x"))
scene.render.resolution_x, scene.render.resolution_y = w, h
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(OUT, "frames", "f_")
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)

# ---- body parts: BVH offsets live in the parent frame, so draw parent.head -> bone.head
RADIUS = {"Hips": 0.10, "Spine": 0.11, "L_UpLeg": 0.075, "R_UpLeg": 0.075,
          "L_Leg": 0.06, "R_Leg": 0.06, "L_Foot": 0.05, "R_Foot": 0.05,
          "L_Shoulder": 0.055, "R_Shoulder": 0.055, "L_ForeArm": 0.045, "R_ForeArm": 0.045,
          "L_Hand": 0.04, "R_Hand": 0.04, "L_Toe": 0.035, "R_Toe": 0.035, "Head": 0.07}
COLOR = {"Hips": (0.85, 0.55, 0.25, 1), "Spine": (0.9, 0.6, 0.3, 1)}

scene.frame_set(1)
bpy.context.view_layer.update()


def add_segment(parent_pb, name, a, b, radius, color):
    d = b - a
    if d.length < 1e-4:
        return
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=radius, depth=1.0)
    ob = bpy.context.object
    ob.name = "part_" + name
    ob.matrix_world = (Matrix.Translation((a + b) / 2)
                       @ d.to_track_quat("Z", "Y").to_matrix().to_4x4()
                       @ Matrix.Diagonal((1, 1, d.length, 1)))
    ob.color = color
    parent_mat = arm.matrix_world @ parent_pb.matrix @ Matrix.Translation((0, parent_pb.bone.length, 0))
    mw = ob.matrix_world.copy()
    ob.parent, ob.parent_type, ob.parent_bone = arm, "BONE", parent_pb.name
    ob.matrix_parent_inverse = parent_mat.inverted()
    ob.matrix_world = mw


for pb in arm.pose.bones:
    if pb.parent is None:
        continue
    radius = RADIUS.get(pb.name, 0.05)
    a = arm.matrix_world @ pb.parent.head
    b = arm.matrix_world @ pb.head
    add_segment(pb.parent, pb.name, a, b, radius, COLOR.get(pb.name, (0.75, 0.78, 0.82, 1)))

head_pb = arm.pose.bones.get("Head")
if head_pb:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.12)
    head = bpy.context.object
    head.name = "part_skull"
    head.color = (0.95, 0.8, 0.35, 1)
    pos = arm.matrix_world @ head_pb.head
    head.matrix_world = Matrix.Translation(pos)
    mw = head.matrix_world.copy()
    parent_mat = arm.matrix_world @ head_pb.matrix @ Matrix.Translation((0, head_pb.bone.length, 0))
    head.parent, head.parent_type, head.parent_bone = arm, "BONE", head_pb.name
    head.matrix_parent_inverse = parent_mat.inverted()
    head.matrix_world = mw

# ---- ground + camera + light ---------------------------------------------
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
ground = bpy.context.object
ground.color = (0.22, 0.24, 0.27, 1)

cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = 40
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

# auto-frame: put the camera in FRONT of the performer (facing from the feet), or fit the motion
if os.environ.get("NPZ_PATH") and os.path.exists(os.environ["NPZ_PATH"]):
    import numpy as np

    z = np.load(os.environ["NPZ_PATH"])
    names = [str(n) for n in z["joint_names"]]
    all_joints = z["joints3d"]  # (frames, joints, 3), metres, Y-up
    joints = all_joints[0]
    pts_npz = np.array(joints, dtype=float)
    pts_bl = np.array([list(arm.matrix_world @ arm.pose.bones[n].head) for n in names], dtype=float)

    A, B = pts_npz - pts_npz.mean(0), pts_bl - pts_bl.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    R = (U @ np.diag([1.0, 1.0, d]) @ Vt).T  # blender = R @ npz + t
    t = pts_bl.mean(0) - R @ pts_npz.mean(0)

    # facing = ankle -> toe, taken from the FIRST frames only, then the camera stays fixed
    def joint_seq(*candidates):
        for name in candidates:
            if name in names:
                return all_joints[:, names.index(name)]
        raise KeyError(f"none of {candidates} in npz joint names")

    ank = joint_seq("L_Ankle", "L_Foot")   # SMPL-X "L_Ankle" (ankle); MediaPipe "L_Foot" is the ankle
    toe = joint_seq("L_Toe", "L_Foot")     # MediaPipe "L_Toe" (foot index); SMPL-X "L_Foot" is the toe
    if np.linalg.norm(toe - ank) < 1e-6:
        raise ValueError("ankle and toe joints resolved to the same point - check joint names")
    n0 = min(10, len(all_joints))
    fwd = np.mean((toe[:n0] - ank[:n0])[:, [0, 2]], axis=0)
    fwd = fwd / max(1e-9, np.linalg.norm(fwd))

    body = all_joints[:, :, :].reshape(-1, 3).mean(0)
    height = float(all_joints[:, :, 1].max() - all_joints[:, :, 1].min())
    dist = max(5.0, height * 2.9)
    # 3/4 view: rotate the front direction by CAM_ANGLE degrees about the vertical axis
    ang = np.radians(float(os.environ.get("CAM_ANGLE", "45")))
    side = 1.0 if os.environ.get("CAM_SIDE", "left") == "left" else -1.0
    c, s = float(np.cos(ang)), float(np.sin(ang)) * side
    dir_x = c * fwd[0] - s * fwd[1]
    dir_z = s * fwd[0] + c * fwd[1]
    cam_npz = body + np.array([dir_x * dist, 0.45 * height, dir_z * dist])
    cam.location = Vector(R @ cam_npz + t)
    target = Vector(R @ (body + np.array([0.0, 0.05, 0.0])) + t)
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.lens = 50.0
    print(f"FACING(first {n0}) fwd=({fwd[0]:.2f},{fwd[1]:.2f}) +{np.degrees(ang) * side:.0f}deg "
          f"dir=({dir_x:.2f},{dir_z:.2f}) camera npz={np.round(cam_npz, 2)} "
          f"blender={tuple(round(v, 2) for v in cam.location)} dist={dist:.1f} (static)", flush=True)
else:
    pts = []
    step = max(1, last // 40)
    for f in range(1, last + 1, step):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        for pb in arm.pose.bones:
            pts.append(arm.matrix_world @ pb.head)
            pts.append(arm.matrix_world @ pb.tail)
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    center = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2))
    height = max(max(zs) - min(zs), 1.2) + 0.25  # + skull radius
    width = max(max(xs) - min(xs), 1.0) + 0.25
    dist = max(height / 0.55, width / 0.95) * 1.35
    cam.location = center + Vector((0.28 * dist, -dist, 0.05 * dist))
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    print(f"FRAME center={tuple(round(v, 2) for v in center)} h={height:.2f} w={width:.2f} dist={dist:.2f}",
          flush=True)
scene.frame_set(1)

light = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "SUN"))
light.data.energy = 3.0
light.rotation_euler = (0.9, 0.2, 0.6)
bpy.context.collection.objects.link(light)

# ---- render --------------------------------------------------------------
print(f"CHECK bones={len(arm.pose.bones)} frames=1-{last} res={w}x{h} fps={FPS}", flush=True)
bpy.ops.render.render(animation=True)
print("RENDER DONE", flush=True)
