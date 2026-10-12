"""Build the workspace forearm model from MyoArm and save it as a pinned, editable XML snapshot.

    myogen build-model            # -> models/myoforearm_r.xml (+ models/meshes/)

Why not MyoHand: a proximal electrode ring (NinaPro DB1/DB5 placement, at the radio-humeral joint) sits over
supinator, brachioradialis and anconeus bellies, and supination is driven by the biceps. MyoHand has none of
these, so its proximal EMG is missing large sources and supination needs reserve torques. This model keeps:

  * joints: elbow flexion, pronation/supination, wrist (2), thumb (4), fingers (16)   -> 24 DoF
  * muscles: the 39 MyoHand muscles + ANC, SUP, BIClong, BICshort, BRA, BRD, TRIlong, TRIlat, TRImed -> 48
  * the humerus is welded with the arm by the side (seated posture; shoulder muscles removed)

The shoulder pose matters because TRIlong and BIClong cross the shoulder: MyoSim's hand-preview pose (90 deg
elevation) stretches TRIlong to 1.2-1.4 optimal lengths, i.e. a 7-9 Nm passive elbow-extension torque that the
elbow flexors would have to fight permanently (tonic BRA/BRD EMG at rest). SHOULDER_POSE was chosen by a grid
search minimising total passive muscle force over elbow flexion 0.4-2.0 rad (~3x less than the preview pose).

The elbow is kept so its angle can be varied per session (limb-position effect: elbow flexors are active,
and visible to a proximal ring, whenever the forearm is held against gravity).
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

from .config import PACKAGE

log = logging.getLogger(__name__)
OUT_DIR = PACKAGE / "models"           # packaged with myogen
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

    bake_pose(spec, SHOULDER_POSE)       # weld the shoulder in a seated, arm-by-the-side posture
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


def main():
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
    OUT_XML.write_text(xml)
    check = mujoco.MjModel.from_xml_path(str(OUT_XML))
    assert (check.nq, check.nu) == (model.nq, model.nu)
    log.info(f"wrote {OUT_XML}: {model.nq} DoF, {model.nu} muscles, {len(spec.meshes)} meshes")

