"""Rebuild the packaged forearm model from MyoArm: python src/myogen/models/build_model.py (needs myo-sim)
-> src/myogen/models/myoforearm_r.xml + meshes/.

Compared to MyoHand it keeps the elbow and the proximal forearm muscles a NinaPro-style armband sits over
(ANC, SUP, BIC, BRA, BRD, TRI): 24 DoF, 48 muscles. The shoulder is welded arm-by-the-side (SHOULDER_POSE, chosen
to minimise passive muscle force over the elbow range); MyoSim's preview pose would stretch TRIlong into a
permanent 7-9 Nm passive elbow torque.
"""
import logging
import re
import shutil
from pathlib import Path

import mujoco
from myo_sim.build.compose import build_right_arm_spec
from myo_sim.build.hand import (
    RIGHT_HAND_ACTUATORS,
    RIGHT_HAND_REMOVED_JOINTS,
    add_side_suffix,
    apply_joint_pose,
    bake_current_body_poses,
    fix_up_welded_phantom_inertia,
    tendon_wrap_geom_names,
)

log = logging.getLogger(__name__)
OUT_DIR = Path(__file__).resolve().parent
OUT_XML = OUT_DIR / "myoforearm_r.xml"

FOREARM_EXTRA_ACTUATORS = frozenset({"ANC", "SUP", "BIClong", "BICshort", "BRA", "BRD", "TRIlong", "TRIlat", "TRImed"})
KEEP_ACTUATORS = RIGHT_HAND_ACTUATORS | FOREARM_EXTRA_ACTUATORS
REMOVED_JOINTS = RIGHT_HAND_REMOVED_JOINTS - {"elbow_flexion_r"}
SHOULDER_POSE = {"elv_angle_r": 0.0, "shoulder_elv_r": 0.1, "shoulder_rot_r": 0.0}


def bake_pose(spec, pose):
    model = spec.compile()
    data = mujoco.MjData(model)
    apply_joint_pose(data, model, pose, "r")
    mujoco.mj_forward(model, data)
    bake_current_body_poses(spec, model, data)


def build_spec():
    spec = build_right_arm_spec()
    spec.modelname = "myoforearm_r"
    spec.compiler.balanceinertia = True
    side = "r"

    for actuator in list(spec.actuators):
        if actuator.name not in KEEP_ACTUATORS:
            spec.delete(actuator)
    tendon_name_map = {a.target: add_side_suffix(a.target, side) for a in spec.actuators}
    for tendon in list(spec.tendons):
        if tendon.name not in tendon_name_map:
            spec.delete(tendon)
        else:
            tendon.name = tendon_name_map[tendon.name]
    wrapped = tendon_wrap_geom_names(spec)
    for geom in list(spec.geoms):
        if geom.group == 3 and geom.name not in wrapped:
            spec.delete(geom)
    for actuator in spec.actuators:
        actuator.target = tendon_name_map[actuator.target]
        actuator.name = add_side_suffix(actuator.name, side)

    bake_pose(spec, SHOULDER_POSE)
    for eq in list(spec.equalities):
        if eq.name1 in REMOVED_JOINTS or eq.name2 in REMOVED_JOINTS:
            spec.delete(eq)
    for name in REMOVED_JOINTS:
        joint = spec.joint(name)
        if joint is not None:
            spec.delete(joint)
    fix_up_welded_phantom_inertia(spec, side)
    for joint in spec.joints:
        joint.name = add_side_suffix(joint.name, side)

    return spec


def build():
    """Write the model XML and its meshes into the package."""
    spec = build_spec()
    model = spec.compile()
    OUT_DIR.mkdir(exist_ok=True)
    xml = spec.to_xml()
    mesh_out = OUT_DIR / "meshes"
    mesh_out.mkdir(exist_ok=True)
    src_dir = Path(spec.meshdir) if spec.meshdir else Path(".")
    for mesh in spec.meshes:
        src = Path(mesh.file) if Path(mesh.file).is_absolute() else src_dir / mesh.file
        shutil.copy(src, mesh_out / src.name)
        xml = xml.replace(f'file="{mesh.file}"', f'file="{src.name}"')
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="meshes"', xml)
    xml = re.sub(r' texturedir="[^"]*"', '', xml)               # only builtin textures: no local paths
    OUT_XML.write_text(xml)
    check = mujoco.MjModel.from_xml_path(str(OUT_XML))
    assert (check.nq, check.nu) == (model.nq, model.nu)
    log.info(f"wrote {OUT_XML}: {model.nq} DoF, {model.nu} muscles, {len(spec.meshes)} meshes")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
