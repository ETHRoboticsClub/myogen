# Rebuild the forearm muscle anatomy from BodyParts3D (docs/anatomy.md):
#   python src/myogen/models/build_anatomy.py <unzipped isa_BP3D_4.0_obj_99 folder>
# -> src/myogen/models/forearm_anatomy.npz, docs/anatomy/slices.png, docs/anatomy/levels.csv
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import trimesh

from myogen.simulation.musculoskeletal import ForearmModel

OUT_DIR = Path(__file__).resolve().parent
OUT_NPZ = OUT_DIR / "forearm_anatomy.npz"
DOCS = OUT_DIR.parents[2] / "docs" / "anatomy"
SOURCE = "BodyParts3D 4.0 (isa, 99), (c) The Database Center for Life Science, CC BY-SA 2.1 Japan"

# BodyParts3D element files (right side) of each model muscle; split: finger compartments, see Atlas.split.
BONES = {"radius": ["FJ3349"], "ulna": ["FJ3391"]}
MUSCLES = {
    "BRD": ["FJ1487"], "ECRL": ["FJ1490"], "ECRB": ["FJ1489"], "ECU": ["FJ1472", "FJ1517"],
    "FCR": ["FJ1496"], "FCU": ["FJ1473", "FJ1518"], "PL": ["FJ1502"], "PT": ["FJ1474", "FJ1516"],
    "PQ": ["FJ1503"], "FDS": ["FJ1475", "FJ1499"], "FDP": ["FJ1497"], "EDC": ["FJ1492"], "EDM": ["FJ1491"],
    "EIP": ["FJ1493"], "EPL": ["FJ1495"], "EPB": ["FJ1494"], "FPL": ["FJ1498"], "APL": ["FJ1484"],
    "SUP": ["FJ1505"], "ANC": ["FJ1485"],
}
FINGERS = ("2", "3", "4", "5")              # radial -> ulnar
LEVELS = np.arange(-0.15, 1.051, 0.025)     # along the radius: 0 head, 1 styloid tip
GRID_MM = 1.0                               # in-plane voxel size
SLAB_MM = 15.0                              # vertices this close to a slice bound its extent
POINTS_PER_MUSCLE = 400                     # stored volume samples per model muscle
ENVELOPE_BINS = 36                          # angular bins of the tissue envelope (normalises depth)
FIT_LEVELS = np.arange(0.1, 0.91, 0.1)      # bone centres compared with the model (check)
FIGURE_LEVELS = (0.15, 0.3, 0.5, 0.7)
FLEXORS = ("FCR", "PL", "FCU")               # wrist flexors and extensors: centres reported as a check
EXTENSORS = ("ECRL", "ECRB", "ECU")
ANCHOR_LEVELS = (0.1, 0.5)                    # where the bands sit


def winding_inside(mesh: trimesh.Trimesh, points: np.ndarray, chunk: int = 1500) -> np.ndarray:
    """Generalised winding number > 0.5: robust to the holes of the reduced meshes."""
    tri = mesh.triangles
    inside = np.zeros(len(points), bool)
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk, None, None, :]
        a, b, c = (tri[None, :, k, :] - p[:, :, 0] for k in range(3))
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        num = np.einsum("ntk,ntk->nt", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ntk,ntk->nt", a, b) * lc + np.einsum("ntk,ntk->nt", a, c) * lb
               + np.einsum("ntk,ntk->nt", b, c) * la)
        inside[i:i + chunk] = np.abs(2 * np.arctan2(num, den).sum(1) / (4 * np.pi)) > 0.5
    return inside


class Atlas:
    """Muscle volumes of BodyParts3D as voxels in a forearm frame: s along the radius (0 head, 1 styloid), angle
    and depth fraction (radius over the tissue envelope at that angle) around the centre of the cross-section."""

    def __init__(self, folder: Path) -> None:
        files = {p.stem: p for p in folder.rglob("*.obj")}
        load = lambda ids: [trimesh.load(files[i], force="mesh", process=True) for i in ids]   # noqa: E731
        self.bones = {k: load(v) for k, v in BONES.items()}
        self.meshes = {k: load(v) for k, v in MUSCLES.items()}
        radius = self.bones["radius"][0].vertices
        bone_pts = np.vstack([m.vertices for ms in self.bones.values() for m in ms])
        centre = bone_pts.mean(0)
        axis = np.linalg.svd(bone_pts - centre, full_matrices=False)[2][0]
        axis *= np.sign(axis @ (radius[np.argmin(radius[:, 2])] - centre))        # towards the wrist (styloid)
        s = (radius - centre) @ axis
        self.head = centre + s.min() * axis
        self.length = float(s.max() - s.min())
        self.axis = axis
        # Anatomical position (supinated): u lateral = radial (BodyParts3D -x on the right), v = axis x u, dorsal on
        # a right forearm (checked against the extensors in Placement). Angles as myogen's: 0 radial, +90 dorsal.
        lateral = np.array([-1.0, 0.0, 0.0])
        u = lateral - (lateral @ axis) * axis
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(axis, self.u)
        self.voxels = {k: self._voxelise(ms) for k, ms in {**self.meshes, **self.bones}.items()}
        self._normalise()

    def _plane_point(self, level: float) -> np.ndarray:
        return self.head + level * self.length * self.axis

    def _voxelise(self, meshes: list[trimesh.Trimesh]) -> np.ndarray:
        """(N, 3) voxel centres (level, u mm, v mm) inside any of the meshes."""
        out = []
        for mesh in meshes:
            rel = mesh.vertices - self.head
            lv = rel @ self.axis / self.length
            uu, vv = rel @ self.u, rel @ self.v
            for level in LEVELS[(LEVELS >= lv.min()) & (LEVELS <= lv.max())]:
                near = np.abs(lv - level) * self.length < SLAB_MM        # the slice lies within these vertices
                if not near.any():
                    continue
                gu = np.arange(np.floor(uu[near].min()), uu[near].max() + GRID_MM, GRID_MM)
                gv = np.arange(np.floor(vv[near].min()), vv[near].max() + GRID_MM, GRID_MM)
                U, V = np.meshgrid(gu, gv, indexing="ij")
                pts = self._plane_point(level) + U.reshape(-1, 1) * self.u + V.reshape(-1, 1) * self.v
                inside = winding_inside(mesh, pts)
                out.append(np.column_stack([np.full(inside.sum(), level), U.ravel()[inside], V.ravel()[inside]]))
        vox = np.vstack(out) if out else np.zeros((0, 3))
        return np.unique(vox, axis=0)

    def _normalise(self) -> None:
        """Per level: centre of all tissue, envelope radius per angle; every voxel gets (level, angle, depth)."""
        tissue = np.vstack(list(self.voxels.values()))
        self.centre, self.envelope = {}, {}
        edges = np.linspace(-np.pi, np.pi, ENVELOPE_BINS + 1)
        for level in LEVELS:
            sl = tissue[np.isclose(tissue[:, 0], level)][:, 1:]
            if len(sl) < 50:
                continue
            c = sl.mean(0)
            ang = np.arctan2(sl[:, 1] - c[1], sl[:, 0] - c[0])
            r = np.hypot(*(sl - c).T)
            env = np.array([np.percentile(r[(ang >= a0) & (ang < a1)], 98) if np.any((ang >= a0) & (ang < a1))
                            else np.nan for a0, a1 in zip(edges[:-1], edges[1:])])
            env = np.where(np.isnan(env), np.nanmean(env), env)
            env = (np.roll(env, 1) + 2 * env + np.roll(env, -1)) / 4     # smooth around the circle
            self.centre[round(level, 4)], self.envelope[round(level, 4)] = c, env
        self.polar = {k: self._to_polar(v) for k, v in self.voxels.items()}

    def _to_polar(self, vox: np.ndarray) -> np.ndarray:
        """(N, 3) level, angle (rad, u = 0 towards v), depth fraction r / envelope(angle)."""
        out = np.full((len(vox), 3), np.nan)
        bins = np.linspace(-np.pi, np.pi, ENVELOPE_BINS + 1)
        for i, (level, uu, vv) in enumerate(vox):
            key = round(level, 4)
            if key not in self.centre:
                continue
            c = self.centre[key]
            ang = np.arctan2(vv - c[1], uu - c[0])
            env = np.interp(ang, (bins[:-1] + bins[1:]) / 2, self.envelope[key], period=2 * np.pi)
            out[i] = level, ang, np.hypot(uu - c[0], vv - c[1]) / env
        return out[~np.isnan(out[:, 0])]

    def bone_angles(self, level: float) -> tuple[float, float] | None:
        """Angles of the radius and ulna centres at a level (atlas frame)."""
        out = []
        for bone in ("radius", "ulna"):
            p = self.voxels[bone]
            sl = p[np.isclose(p[:, 0], level)]
            if not len(sl) or round(level, 4) not in self.centre:
                return None
            d = sl[:, 1:].mean(0) - self.centre[round(level, 4)]
            out.append(np.arctan2(d[1], d[0]))
        return out[0], out[1]


class Placement:
    """The atlas in the model's forearm frame: both anatomical, supinated (0 radial = lateral, +90 dorsal =
    posterior), so no rotation. The angles hold at the model's default pose, q_ref (its distal radius and ulna sit
    where the atlas has them); posture moves them with the model's bellies (EMGModel). Reported as checks: the
    bone centres of both, and the wrist flexors' and extensors' centres (palmar-ulnar and dorsal-radial in a
    supinated forearm, not at -90 / +90)."""

    def __init__(self, atlas: Atlas, model: ForearmModel) -> None:
        self.atlas, self.model = atlas, model
        dorsal, palmar = (np.angle(np.mean(np.exp(1j * atlas.polar[n][:, 1]))) for n in ("EDC", "FDS"))
        if not (np.sin(dorsal) > 0.5 and np.sin(palmar) < -0.5):
            raise ValueError(f"atlas frame: extensors at {np.degrees(dorsal):.0f} deg, flexors at "
                             f"{np.degrees(palmar):.0f} deg; expected dorsal (+90) and palmar (-90)")
        self.q_ref = model.default_qpos.copy()
        self.offset = 0.0
        self.flexors_deg, self.extensors_deg = (float(np.degrees(self.model_angle(self.centre_angle(names))))
                                                for names in (FLEXORS, EXTENSORS))

    def centre_angle(self, names: tuple[str, ...]) -> float:
        """Circular mean angle (atlas frame) of these muscles over the band region."""
        pts = np.vstack([self.atlas.polar[n] for n in names])
        pts = pts[(pts[:, 0] >= ANCHOR_LEVELS[0]) & (pts[:, 0] <= ANCHOR_LEVELS[1])]
        return float(np.angle(np.mean(np.exp(1j * pts[:, 1]))))

    def bone_check(self) -> list[tuple[float, str, float, float]]:
        """(level, bone, atlas angle, model angle at q_ref) in degrees."""
        rows = []
        for level in FIT_LEVELS:
            a, mdl = self.atlas.bone_angles(level), self.model_bone_angles(self.q_ref, level)
            if a is not None and mdl is not None:
                rows += [(level, bone, np.degrees(self.model_angle(a[k])), np.degrees(mdl[k])) for k, bone in
                         enumerate(("radius", "ulna"))]
        return rows

    def model_bone_angles(self, q: np.ndarray, level: float) -> tuple[float, float] | None:
        m = self.model.m
        d = mujoco.MjData(m)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        origin, axes = self.model.forearm_frame(d)
        pts = {}
        for bone in ("radius", "ulna"):
            g = m.geom(f"{bone}_r").id
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts[bone] = ((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T) - origin) @ axes.T
        r = pts["radius"][:, 2]
        z = r.min() + level * (r.max() - r.min())
        out = []
        for bone in ("radius", "ulna"):
            p = pts[bone]
            sel = np.abs(p[:, 2] - z) < 0.006
            if sel.sum() < 3:
                return None
            out.append(np.arctan2(*p[sel, :2].mean(0)[::-1]))
        return out[0], out[1]

    def model_angle(self, atlas_angle: np.ndarray) -> np.ndarray:
        return np.angle(np.exp(1j * (atlas_angle + self.offset)))


def split(points: np.ndarray, name: str) -> dict[str, np.ndarray]:
    """Finger compartments, per level: FDP / EDC radial -> ulnar by angle (index on the radial side); FDS in two
    layers, middle and ring superficial, index and little deep (each layer radial -> ulnar)."""
    out = {f: [] for f in FINGERS}
    for level in np.unique(points[:, 0]):
        sl = points[points[:, 0] == level]
        order = np.argsort(np.abs(sl[:, 1]))          # angular distance from the radial border (model frame)
        if name == "FDS":
            deep = sl[:, 2] < np.median(sl[:, 2])
            for layer, (radial, ulnar) in ((~deep, ("3", "4")), (deep, ("2", "5"))):
                part = sl[layer][np.argsort(np.abs(sl[layer, 1]))]
                half = len(part) // 2
                out[radial].append(part[:half])
                out[ulnar].append(part[half:])
        else:
            for f, part in zip(FINGERS, np.array_split(sl[order], 4)):
                out[f].append(part)
    return {f: np.vstack(v) for f, v in out.items()}


def model_points(atlas: Atlas, place: Placement) -> dict[str, np.ndarray]:
    """(N, 3) level, angle (model frame: 0 radial, +90 dorsal), depth fraction per model muscle."""
    rng = np.random.default_rng(0)
    out = {}
    for name, pts in atlas.polar.items():
        if name in BONES:
            continue
        pts = pts.copy()
        pts[:, 1] = place.model_angle(pts[:, 1])
        parts = split(pts, name) if name in ("FDS", "FDP", "EDC") else {"": pts}
        for suffix, p in parts.items():
            keep = rng.choice(len(p), min(POINTS_PER_MUSCLE, len(p)), replace=False)
            out[f"{name}{suffix}_r"] = p[keep]
    return out


def line_of_action(model: ForearmModel, q: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """The model's own belly points (level along the radius, angle) at q, for the comparison figure."""
    m = model.m
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    pts = model.belly_points(12, d)
    origin, axes = model.forearm_frame(d)
    g = m.geom("radius_r").id
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    rz = (((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T) - origin) @ axes.T)[:, 2]
    out = {}
    for n in names:
        p = pts[model.muscle_names.index(n)]
        out[n] = np.column_stack([(p[:, 2] - rz.min()) / (rz.max() - rz.min()), np.arctan2(p[:, 1], p[:, 0]),
                                  np.hypot(p[:, 0], p[:, 1])])
    return out


def write_report(atlas: Atlas, place: Placement, points: dict[str, np.ndarray], model: ForearmModel) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = line_of_action(model, place.q_ref, list(points))
    rows = []
    for name, p in points.items():
        for level in FIGURE_LEVELS:
            sl = p[np.abs(p[:, 0] - level) < 0.026]
            ln = lines[name]
            near = np.abs(ln[:, 0] - level) < 0.08
            old = np.degrees(np.angle(np.mean(np.exp(1j * ln[near, 1])))) if near.any() else np.nan
            if len(sl):
                rows.append([name, level, round(np.degrees(np.angle(np.mean(np.exp(1j * sl[:, 1])))), 1),
                             round(float(np.mean(sl[:, 2])), 3), len(sl), round(old, 1)])
    with open(DOCS / "levels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {SOURCE}"])
        w.writerow(["muscle", "level", "angle_deg", "depth_fraction", "samples", "line_of_action_angle_deg"])
        w.writerows(rows)

    fig, axs = plt.subplots(1, len(FIGURE_LEVELS), figsize=(5 * len(FIGURE_LEVELS), 5.6),
                            subplot_kw={"projection": "polar"})
    cmap = plt.get_cmap("tab20")
    names = sorted({n[:-2].rstrip("2345") if n[:3] in ("FDS", "FDP", "EDC") else n[:-2] for n in points})
    colour = {n: cmap(i % 20) for i, n in enumerate(names)}
    for ax, level in zip(axs, FIGURE_LEVELS):
        for bone in BONES:
            p = atlas.polar[bone]
            sl = p[np.isclose(p[:, 0], level)]
            ax.scatter(place.model_angle(sl[:, 1]), sl[:, 2], s=3, c="0.6", lw=0)
        p_all = {n: atlas.polar[n] for n in MUSCLES}
        for n, p in p_all.items():
            sl = p[np.isclose(p[:, 0], level)]
            if not len(sl):
                continue
            ang = place.model_angle(sl[:, 1])
            ax.scatter(ang, sl[:, 2], s=3, color=colour[n], lw=0)
            c = np.angle(np.mean(np.exp(1j * ang)))
            ax.text(c, np.mean(sl[:, 2]), n, fontsize=7, ha="center", va="center", weight="bold")
        for n, ln in lines.items():
            near = np.abs(ln[:, 0] - level) < 0.05
            if near.any():
                base = n[:-2].rstrip("2345") if n[:3] in ("FDS", "FDP", "EDC") else n[:-2]
                ax.scatter(ln[near, 1], np.full(near.sum(), 1.08), marker="x", s=14, color=colour[base])
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels([])
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["radial", "dorsal", "ulnar", "palmar"])
        ax.set_title(f"{level:.0%} of the radius from the elbow")
    fig.suptitle("Forearm muscles from BodyParts3D in myogen's forearm frame (looking from the elbow; radius = "
                 "depth fraction). x on the rim: the model's line-of-action angle (old geometry)\n"
                 f"{SOURCE}", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "slices.png", dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("meshes", type=Path, help="unzipped BodyParts3D isa_BP3D_4.0_obj_99 folder")
    args = parser.parse_args()
    model = ForearmModel()
    atlas = Atlas(args.meshes)
    place = Placement(atlas, model)
    points = model_points(atlas, place)
    print(f"wrist flexors at {place.flexors_deg:.0f} deg, extensors at {place.extensors_deg:.0f} deg")
    for level, bone, a, mdl in place.bone_check():
        print(f"{level:.1f} {bone:6s} atlas {a:6.0f}  model {mdl:6.0f} deg")
    np.savez_compressed(OUT_NPZ, names=np.array(list(points)), q_ref=place.q_ref, source=SOURCE,
                        **{f"points_{k}": v.astype(np.float32) for k, v in points.items()})
    write_report(atlas, place, points, model)


if __name__ == "__main__":
    main()
