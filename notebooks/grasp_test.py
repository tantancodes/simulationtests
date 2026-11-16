"""Grasp test: does reconstruction error change a pick?

STATUS: in progress, results not validated. The finger pads can push into the
dog's collision hulls, and outcomes change more with gripper settings than with
the reconstruction. Next step is checking the rig on a plain box first.

Floating Panda gripper (no arm, moved directly via a mocap body) picks the
scanned dog. Same script for the reference and side40 meshes.

Usage (from notebooks/):
    python grasp_test.py build      # scale assets + write scene XMLs
    python grasp_test.py run        # run test + control grasps, print table
    mjpython grasp_test.py watch reference test      # watch one grasp in the viewer (Mac needs mjpython)
    mjpython grasp_test.py watch side40 control 4    # last number = slow motion factor
    mjpython grasp_test.py watch reference test 6    # replays on a loop; space = pause, close window to quit
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
import mujoco
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
SRC = HERE / 'assets'                      # v0.5 outputs (20 cm dog)
OUT = HERE / 'assets' / 'grasp'
PANDA = HERE.parent / 'third_party' / 'franka_emika_panda'

SCALE = 0.4          # 20 cm assumed -> 8 cm, so the body fits the 8 cm gripper. Still an assumption.
DENSITY = 1500       # kg/m^3, assumed
DOGS = ['reference', 'side40']

# grasps chosen from hull widths (see README): z = pad height, angle = closing direction in xy
GRASPS = {
    'test (head)':    dict(z=0.060, angle=165),   # ref 3.4 cm vs side40 4.5 cm wide here
    'control (body)': dict(z=0.048, angle=120),   # both 5.02 cm wide here
}
# grasp centre = middle of the dog's cross-section at pad height, measured on the REFERENCE hulls,
# so both dogs get the exact same gripper motion

GRIP_STIFFNESS = 1000   # N/m on the finger tendon; Menagerie default is 100 (see scene_xml)
GRIP_FORCE = 20         # N, max squeeze (real Franka gripper: 20-70 N)
CONDIM = 4              # 3 = sliding friction only; 4 adds twist friction, so a pinch isn't a hinge
TIMESTEP = 0.001        # s (MuJoCo default 0.002)

JITTER = dict(xy=0.003, z=0.003, angle=10)   # +/- for the robustness sweep (m, m, deg)


def build():
    for name in DOGS:
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        trimesh.load(SRC / name / 'visual.obj').apply_scale(SCALE).export(folder / 'visual.obj')
        n = len(list((SRC / name).glob('hull_*.obj')))
        for i in range(n):   # scaling a convex hull keeps it convex, so no need to rerun CoACD
            trimesh.load(SRC / name / f'hull_{i}.obj').apply_scale(SCALE).export(folder / f'hull_{i}.obj')
        (OUT / f'scene_{name}.xml').write_text(scene_xml(name, n))
        print(f'{name}: {n} hulls -> {OUT / f"scene_{name}.xml"}')


def hand_quat(angle_deg):
    """Fingers pointing down, fingers closing along `angle_deg` in the xy plane."""
    down = R.from_euler('x', 180, degrees=True)          # hand's +z (finger direction) -> world -z
    # after `down`, fingers close along world y; spin so they close along angle_deg
    q = (R.from_euler('z', angle_deg - 90, degrees=True) * down).as_quat()  # x y z w
    return np.array([q[3], q[0], q[1], q[2]])           # mujoco order w x y z


def scene_xml(name, n_hulls):
    rel = lambda p: Path('../../..') / p.relative_to(HERE.parent)   # path relative to meshdir
    hand = (PANDA / 'hand.xml').read_text()
    hand = hand.replace('meshdir="assets"', 'meshdir="../../../third_party/franka_emika_panda/assets"')

    dog_assets = f'<mesh name="dog_vis" file="{rel(OUT / name / "visual.obj")}" inertia="shell"/>' + ''.join(
        f'<mesh name="dog_h{i}" file="{rel(OUT / name / f"hull_{i}.obj")}"/>' for i in range(n_hulls))
    dog_geoms = ('<geom type="mesh" mesh="dog_vis" contype="0" conaffinity="0" density="0" rgba=".6 .5 .4 1"/>'
                 + ''.join(f'<geom type="mesh" mesh="dog_h{i}" density="{DENSITY}" friction="0.8" condim="{CONDIM}" '
                           f'rgba="1 0 0 0.5" group="3"/>' for i in range(n_hulls)))
    world = f'''
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="0.5 0.5 .01" rgba=".8 .8 .8 1"/>
    <body name="dog" pos="0 0 0.001"><freejoint name="dog"/>{dog_geoms}</body>
    <body name="target" mocap="true" pos="0 0 0.3"/>
    '''
    hand = hand.replace('</asset>', dog_assets + '</asset>')
    hand = hand.replace('<worldbody>', '<worldbody>' + world)
    hand = hand.replace('<body name="hand" childclass="panda" quat="0 0 0 1">',
                        '<body name="hand" childclass="panda" pos="0 0 0.3"><freejoint name="hand"/>')
    # the hand follows the mocap target through a weld (a stiff spring), like a perfect arm
    hand = hand.replace('<equality>', '<equality><weld body1="hand" body2="target"/>')
    # Menagerie's gripper is a position servo with stiffness 100 N/m: on a 4 cm object it only
    # squeezes ~2 N. The real Franka gripper does 20-70 N. Stiffen it so a 4 cm grip is ~20 N.
    # A stiff servo alone pushes the pads into the hulls, so also cap the force at GRIP_FORCE.
    kp = GRIP_STIFFNESS
    hand = hand.replace('forcerange="-100 100" ctrlrange="0 255"\n      gainprm="0.01568627451 0 0" biasprm="0 -100 -10"',
                        f'forcerange="-{GRIP_FORCE} {GRIP_FORCE}" ctrlrange="0 255"\n      '
                        f'gainprm="{0.04 * kp / 255} 0 0" biasprm="0 -{kp} -{kp / 10}"')
    # smaller timestep = stiffer, less mushy contacts
    hand = hand.replace('<option integrator="implicitfast"/>', f'<option integrator="implicitfast" timestep="{TIMESTEP}"/>')
    return hand


def pose_hand(model, data, pos, quat):
    data.mocap_pos[0], data.mocap_quat[0] = pos, quat
    j = model.joint('hand').qposadr[0]
    data.qpos[j:j + 3], data.qpos[j + 3:j + 7] = pos, quat


def hull_points(model, data, body):
    """All collision-hull vertices of a body, in world coordinates."""
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not model.geom_contype[g]:
            continue
        m = model.geom_dataid[g]
        s, n = model.mesh_vertadr[m], model.mesh_vertnum[m]
        pts.append(data.geom_xpos[g] + model.mesh_vert[s:s + n] @ data.geom_xmat[g].reshape(3, 3).T)
    return np.vstack(pts)


def grasp_centre(z):
    """Middle of the reference dog's hull cross-section at height z (dog resting at the origin)."""
    model = mujoco.MjModel.from_xml_path(str(OUT / 'scene_reference.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    P = hull_points(model, data, model.body('dog').id)
    band = P[np.abs(P[:, 2] - z) < 0.004]
    return band[:, :2].mean(0)


def smooth(a):
    return a * a * (3 - 2 * a)   # ease in/out, so the hand never jerks to a stop


def run_grasp(name, z, angle, centre, viewer=None, model=None, data=None):
    """One pick. Returns whether the dog was still held at the end, and how it moved."""
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(OUT / f'scene_{name}.xml'))
        data = mujoco.MjData(model)
    dog, hand = model.body('dog').id, model.body('hand').id
    pads = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] != hand and model.body(model.geom_bodyid[g]).name.endswith('finger')
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    q = hand_quat(angle)

    # measure hand origin -> pad centre instead of trusting a hand-typed number
    pose_hand(model, data, [0, 0, 0.3], q)
    mujoco.mj_forward(model, data)
    pad_offset = data.xpos[hand][2] - np.mean([data.geom_xpos[g][2] for g in pads])

    # 1. dog settles with the open hand far above
    data.ctrl[0] = 255                                   # 255 = open, 0 = closed
    while data.time < 0.5:
        mujoco.mj_step(model, data)
    start_z = data.xipos[dog][2]      # centre-of-mass height (body origin is the dog's bottom, which rises if it flips)
    start_rot = data.xmat[dog].reshape(3, 3).copy()

    # 2. scripted motion: (end time, hand target, gripper command)
    at = np.array([centre[0], centre[1], z + pad_offset])
    above, lifted = at + [0, 0, 0.08], at + [0, 0, 0.10]
    plan = [(1.0, above, 255), (2.0, at, 255), (2.8, at, 0), (3.8, lifted, 0), (5.3, lifted, 0)]
    pose_hand(model, data, above, q)
    t0, p0, dropped_at, was_lifted = data.time, above, None, False
    for t_end, p_end, grip in plan:
        while data.time < t_end:
            data.mocap_pos[0] = p0 + smooth(min(1, (data.time - t0) / (t_end - t0))) * (p_end - p0)
            data.ctrl[0] = grip
            mujoco.mj_step(model, data)
            if viewer:
                viewer.sync()
            lift = data.xipos[dog][2] - start_z
            if lift > 0.03:
                was_lifted = True
            if was_lifted and dropped_at is None and lift < 0.03:
                dropped_at = data.time      # was lifted, now back near the floor
        if t_end == 3.8:
            peak = data.xipos[dog][2] - start_z
        t0, p0 = t_end, p_end

    j1, j2 = (model.joint(f'finger_joint{i}').qposadr[0] for i in (1, 2))
    final_lift = data.xipos[dog][2] - start_z
    return dict(
        held=final_lift > 0.08,
        peak_lift_cm=peak * 100,
        final_lift_cm=final_lift * 100,
        dropped_at=dropped_at,
        finger_gap_cm=(data.qpos[j1] + data.qpos[j2]) * 100,
        tilt_deg=np.degrees(np.linalg.norm(R.from_matrix(start_rot.T @ data.xmat[dog].reshape(3, 3)).as_rotvec())),
        mass_g=model.body_subtreemass[dog] * 1000,
    )


def run():
    """Nominal grasp once per dog, then a jittered sweep (same jitters for both dogs)."""
    rng = np.random.default_rng(0)
    jitters = [np.zeros(4)] + [rng.uniform(-1, 1, 4) for _ in range(26)]   # 27 trials
    for gname, g in GRASPS.items():
        c0 = grasp_centre(g['z'])
        print(f'\n{gname}: pads at z = {g["z"]*100:.1f} cm, closing at {g["angle"]} deg, centre ({c0[0]*100:+.1f}, {c0[1]*100:+.1f}) cm')
        for name in DOGS:
            rs = []
            for j in jitters:
                c = c0 + j[:2] * JITTER['xy']
                rs.append(run_grasp(name, g['z'] + j[2] * JITTER['z'], g['angle'] + j[3] * JITTER['angle'], c))
            nom = rs[0]
            gaps = [r['finger_gap_cm'] for r in rs if r['held']]
            print(f'  {name:9} nominal: {"HELD" if nom["held"] else "DROPPED"} (peak lift {nom["peak_lift_cm"]:.1f} cm, '
                  f'gap {nom["finger_gap_cm"]:.2f} cm, tilt {nom["tilt_deg"]:.0f} deg) | '
                  f'jitter sweep: held {sum(r["held"] for r in rs)}/{len(rs)}'
                  + (f', grip gap {np.mean(gaps):.2f} cm' if gaps else ''))
    return


def watch():
    """Replay the nominal grasp in the viewer, with collision hulls and contact forces shown."""
    import time
    import mujoco.viewer
    name = sys.argv[2]                                        # reference / side40
    gname = next(k for k in GRASPS if k.startswith(sys.argv[3]))   # test / control
    slow = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].replace('.', '', 1).isdigit() else 2.0
    g = GRASPS[gname]
    model = mujoco.MjModel.from_xml_path(str(OUT / f'scene_{name}.xml'))
    data = mujoco.MjData(model)
    paused = [False]
    def on_key(key):
        if key == 32:                      # space bar: pause / resume
            paused[0] = not paused[0]
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
        v.opt.geomgroup[3] = 1                                   # collision hulls (red / grey)
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1    # contact points
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1    # contact force arrows

        class Paced:   # run_grasp calls .sync() every physics step; redraw every 10 and sleep
            n = 0
            def sync(self):
                if not v.is_running():
                    raise SystemExit   # window closed mid-replay
                while paused[0] and v.is_running():   # frozen, but you can still rotate/zoom
                    v.sync(); time.sleep(0.05)
                self.n += 1
                if self.n % 10 == 0:
                    v.sync()
                    time.sleep(10 * model.opt.timestep * slow)

        centre = grasp_centre(g['z'])
        first = True
        while v.is_running():   # replay forever; close the window to exit
            mujoco.mj_resetData(model, data)       # back to t = 0, dog on the floor, hand up top
            r = run_grasp(name, g['z'], g['angle'], centre, viewer=Paced(), model=model, data=data)
            if first:
                print(f'{gname} | {name}: ' + ', '.join(
                    f'{k}={v_:.2f}' if isinstance(v_, (float, np.floating)) else f'{k}={bool(v_) if isinstance(v_, np.bool_) else v_}'
                    for k, v_ in r.items()))
                first = False
            end = time.time() + 2                  # pause on the final pose, then go again
            while v.is_running() and time.time() < end:
                v.sync(); time.sleep(0.05)


USAGE = """usage (from notebooks/, venv active):
  python grasp_test.py build
  python grasp_test.py run
  mjpython grasp_test.py watch reference test
  mjpython grasp_test.py watch side40 control 4"""

if __name__ == '__main__':
    commands = {'build': build, 'run': run, 'watch': watch}
    if len(sys.argv) < 2:
        # plain "python grasp_test.py" (e.g. the VS Code Run button): build if needed, then the sweep
        print(USAGE + '\n\nNo command given, so running the sweep.\n')
        sys.argv.append('run')
    if sys.argv[1] not in commands or (sys.argv[1] == 'watch' and len(sys.argv) < 4):
        sys.exit(USAGE)
    if sys.argv[1] != 'build' and not all((OUT / f'scene_{d}.xml').exists() for d in DOGS):
        print('scene files missing, building them first')
        build()
    try:
        commands[sys.argv[1]]()
    except KeyboardInterrupt:
        pass   # Ctrl+C: just stop

#on startup
# cd ~/Documents/scene2sim/notebooks
# source ../.venv/bin/activate
# mjpython grasp_test.py watch reference test 6 