# Scene2Sim

Scene2Sim turns a 3D scan of a real object into something a physics simulator can use. It takes a photogrammetry mesh, cleans it up, orients it, gives it a size and mass, builds convex collision geometry, and loads it into MuJoCo as a rigid body.

The first experiment ([notebooks/01_drop_test.ipynb](notebooks/01_drop_test.ipynb)) drops a scanned dog figurine onto a floor and checks that it behaves sensibly.

## Setup

Requires Python 3.10+ (developed on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy networkx shapely rtree trimesh coacd mujoco matplotlib jupyter
```

What each package is for:

- `numpy`: arrays and linear algebra (camera-ring PCA)
- `trimesh`: mesh loading, splitting, slicing, export
- `scipy`, `networkx`, `shapely`, `rtree`: optional trimesh dependencies that its mesh operations (splitting into pieces, slicing) rely on
- `coacd`: convex decomposition for collision geometry
- `mujoco`: physics simulation and offscreen rendering
- `matplotlib`: plots
- `jupyter`: running the notebook

## Running

The notebook reads the scan and camera poses from the lab's reconstruction project, which is not in this repo:

```
~/Documents/bd3d-reconstruction-quality/results/gpu-pilot-d8/ref50/raw-mesh.obj
~/Documents/bd3d-reconstruction-quality/results/gpu-pilot-d8/ref50/cameras.csv
```

Change `SCAN_DIR` in the Settings cell if they live elsewhere. Then:

```bash
source .venv/bin/activate
jupyter lab notebooks/01_drop_test.ipynb    # then Run All
```

or headless:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_drop_test.ipynb
```

Generated meshes are written to `notebooks/assets/`. That folder is gitignored because the scan belongs to the lab and needs permission before it can be published.

## Current results

| Step | Result |
|---|---|
| Raw scan | 4 connected pieces (dog + 3 floaters) |
| Base cut | at 0.19 of scan height, just above the platform spike in the height histogram |
| Scale | 0.20 m tall (**assumed**) → 0.174 × 0.247 × 0.200 m |
| Collision | 67 convex pieces (CoACD, threshold 0.08) |
| Mass | 1121 g at 1500 kg/m³ (**assumed** density, solid body) |
| Drop | 0.3 m with a 15°/10° tilt → final tilt 0.0°, stayed upright |
| Sanity check | first contact at 0.238 s vs. √(2h/g) = 0.237 s |

**Assumptions and limitations:**

- The scan has no real-world scale, so the height (0.20 m) and density (1500 kg/m³) are guesses. Mass follows from both.
- The figurine is treated as solid.
- The scan has holes, which CoACD fills in by remeshing.
- Cutting off the scanned base leaves a flat bottom, so landing upright from a small tilt is close to trivial. The result shows the pipeline works end to end. It is not yet a hard test of the simulator.
