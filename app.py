from time import time

import numpy as np
from skimage import draw, filters, exposure, measure
from scipy import ndimage
import os
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash_slicer import VolumeSlicer
import tifffile
import torch
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)
server = app.server
# TODO:
# Optimise code (when to load img etc)
# Good param ranges>

# Add more usefull buttons
# Fix position

# Cleanup code, (format, remove useless stuff)

# Add other graphs (A way to view and compare mask, training slices?)
# Other graph shows score?

# ------------- Define App Layout ---------------------------------------------------
mesh_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("3D mesh representation of the image data and annotation"),
        dcc.Graph(id="graph-helper"),
        #dbc.CardBody([dcc.Graph(id="graph-helper", figure=fig_mesh)]),
        ]
    )
    ]
)

# New button which holds params that can alter the processing of the image
# Check other app, might need to set col + html title 
button_stepsize = dcc.Dropdown(
    id = "set_stepsize",
    options = [{"label": str(i), "value": i}
               for i in range(1, 10)
              ],
    value=3,
)

nav_bar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H3("Covid X-Ray app"),
                                            html.P(
                                                "Exploration and annotation of CT images"
                                            ),
                                        ],
                                        id="app_title",
                                    )
                                ),
                            ],
                            align="center",
                            style={"display": "inline-flex"},
                        )
                    ),
                ]
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
)

# We can put a body in here, the define the body above which holds: buttons and graph
app.layout = html.Div(
    [
        nav_bar,
        dbc.Container(
            [
                dbc.Row([dbc.Col(mesh_card),]),
                dbc.Row([dbc.Col(button_stepsize),]),
            ],
            fluid=True,
        ),
        dcc.Store(id="annotations", data={}),
        dcc.Store(id="occlusion-surface", data={}),
    ],
)

# ------------- Define App Interactivity ---------------------------------------------------
@app.callback(
    Output("graph-helper", "figure"), [Input("set_stepsize", "value")],
)
def create_histo(step_size):
    # Is there  a way to compute this on gpu?
    # Can also put this in top function and call the function every time a param is udpated
    labels_folder_path = 'D:/data/train/kidney_3_sparse/labels'  # Adjust the path accordingly
    label_images = []
    for file in sorted(os.listdir(labels_folder_path))[100:900]:
        if file.endswith(".tif"):
            label_image = tifffile.imread(os.path.join(labels_folder_path, file))
            label_image_downsampled = label_image[::1, ::1]
            label_images.append(torch.from_numpy(label_image_downsampled).float())

    # Combine the first 100 images into one 3D tensor
    all_images = torch.stack(label_images, dim=0)
    print("we get here")
    print(step_size)
    # Extract coordinates of non-zero values (assuming binary segmentation)
    non_zero_coords = all_images.nonzero()
    med_img = non_zero_coords
    med_img = all_images.numpy()

    # Create mesh
    # Step_size affects the level of detail and 200 is threshold. When volume over this threshold we graph it.
    # Marching cube reduces the data complexity. 
    # Without MC you dont have any surface roconstruction or isosurface extraction
    verts, faces, _, _ = measure.marching_cubes(med_img, 200, step_size=step_size)
    x, y, z = verts.T
    i, j, k = faces.T
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
