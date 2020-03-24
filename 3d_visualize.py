import numpy as np
import plotly.graph_objects as go

from ase.visualize import view
from ase.io.cube import read_cube_data

cubefile = "qm9_000001_PBE1PBE_pcS-3.cube"
with open(cubefile) as f:
	data, atoms = read_cube_data(f)

X, Y, Z = np.mgrid[0:6.95:139j, 0:6.95:139j, 0:6.95:139j] 
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=data.flatten(),
    isomin=0,
    isomax=data.max(),
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=21, # needs to be a large number for good volume rendering
    ))
fig.show()  