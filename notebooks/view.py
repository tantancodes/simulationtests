import mujoco, mujoco.viewer
from pathlib import Path

here = Path(__file__).resolve().parent          # the notebooks/ folder
scene = here / 'scene.xml'
print('loading', scene, '| exists:', scene.exists())

model = mujoco.MjModel.from_xml_path(str(scene))  # errors here = problem in the file
mujoco.viewer.launch(model)                       # opens the window with the dog already loaded


## run it with 
## cd ~/Documents/scene2sim/notebooks
##python view.py


