"""Open a saved scene in the MuJoCo viewer.

    python view.py drop        # notebook 01: one dog
    python view.py compare     # notebook 02: reference and side40 side by side
    python view.py side40      # notebook 02: one of the two dogs
    python view.py path/to/any_scene.xml

Viewer controls: double click a body, then ctrl + right drag to push it around.
Press 3 to show/hide collision hulls (the red pieces).
"""
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

here = Path(__file__).resolve().parent          # the notebooks/ folder
scenes = {
    'drop': here / 'assets' / 'scene.xml',
    'compare': here / 'assets' / 'compare.xml',
    'reference': here / 'assets' / 'reference' / 'scene.xml',
    'side40': here / 'assets' / 'side40' / 'scene.xml',
}

name = sys.argv[1] if len(sys.argv) > 1 else 'compare'
scene = scenes.get(name, Path(name))
if not scene.exists():
    sys.exit(f'{scene} not found. Run the notebook that makes it first.')

print('loading', scene)
model = mujoco.MjModel.from_xml_path(str(scene))  # errors here = problem in the file
mujoco.viewer.launch(model)
