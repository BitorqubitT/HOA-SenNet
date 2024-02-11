from time import time

import rasterio
from rasterio.plot import show
from PIL import Image
import matplotlib.pyplot as plt

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

# Change this to more stable version without sliders
# Can reload the whole image every time.

# Check .nii file
# Should i just load 3d matrix the go over it with slider?
# Does it fit in memory?
# other methods?
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

# show picture of liver
liver_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("2D slices of liver"),
        dcc.Graph(id="graph-liver"),
        ]
    )
    ]
)

# show masked of liver
mask_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("2D mask of liver"),
        dcc.Graph(id="graph-liver-msk"),
        ]
    )
    ]
)

mask_card2 = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("2D mask of liver"),
        dcc.Graph(id="graph-liver-msk2"),
        ]
    )
    ]
)

# check effect of resizing
# Check if I can implement this similar to the rest
def pre_load():
    labels_folder_path = 'D:/data/train/kidney_1_dense/images/'  # Adjust the path accordingly
    label_images = []
    for file in sorted(os.listdir(labels_folder_path))[800:950]:
        if file.endswith(".tif"):
            label_image = tifffile.imread(os.path.join(labels_folder_path, file))
            label_image = label_image[0::2, 0::2]
            label_images.append(label_image)
    # Combine the first 100 images into one 3D tensor
    x = np.array(label_images)
    return x

x = pre_load()

# New button which holds params that can alter the processing of the image
# Check other app, might need to set col + html title 
button_stepsize = dcc.Dropdown(
    id = "set_stepsize",
    options = [{"label": str(i), "value": i}
               for i in range(1, 10)
              ],
    value=3,
)

button_threshold = dcc.Dropdown(
    id = "set_threshold",
    options = [{"label": str(i), "value": i}
               for i in range(100, 300, 100)
              ],
    value=200,
)

button_pickslice = dcc.Dropdown(
    id = "set_slice",
    options = [{"label": str(i), "value": i}
               for i in range(1000, 1003, 1)
              ],
    value=1100,
)

slider_liver = dcc.Slider(
    1, 300, 1,
    id = "slice_slider",
    value=200,
    )

# create two buttons for liver pic

# We can put a body in here, the define the body above which holds: buttons and graph
# might want to use storesssssssssssssss

app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row([dbc.Col(html.P(["Choose a step_size:"])),
                         dbc.Col([dbc.Col(button_stepsize),]),
                         dbc.Col(html.P(["Choose a threshold"])),
                         dbc.Col([dbc.Col(button_threshold),])
                         ]),
                dbc.Row([dbc.Col(mesh_card),]),
                dbc.Row([dbc.Col(html.P(["Choose a slice to inspect:"])),
                         dbc.Col([dbc.Col(button_pickslice),])
                        ]),
                dbc.Row([dbc.Col(liver_card),
                         dbc.Col(mask_card)
                        ]),                     
                dbc.Row([dbc.Col(slider_liver),
                         dbc.Col(mask_card2)
                        ]),                     
            ],
            fluid=True,
        ),
    ],
)

# ------------- Define App Interactivity ---------------------------------------------------
@app.callback(
    Output("graph-helper", "figure"),
    [Input("set_stepsize", "value"), Input("set_threshold", "value"), Input("slice_slider", "value")],
)
def create_histo(step_size, threshold, slice_slider):
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
    print(all_images.shape)
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
    verts, faces, _, _ = measure.marching_cubes(med_img, threshold, step_size=step_size)
    x, y, z = verts.T
    i, j, k = faces.T
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))

    # add plane
    # Assuming you have the dimensions of your volume
    volume_shape = all_images.shape

    # Define the height of the horizontal plane
    plane_height = volume_shape[0] // 2  # You can adjust this as needed
    print("slider value is:", slice_slider)
    print("height is:", plane_height)
    # Create grid for the plane
    x_plane, y_plane = np.meshgrid(np.arange(volume_shape[2]), np.arange(volume_shape[1]))
    z_plane = np.full_like(x_plane, plane_height)

    fig.add_trace(go.Surface(z=z_plane, y=y_plane, x=x_plane, opacity=0.5))

    return fig

# Can clean up this code by having 2 outputs in the same callback
# Dont need two functions then
# Cleaner way to share the input

# Check what happens if we load the whole image in one go.
# then only change the slice 
@app.callback(
    Output("graph-liver", "figure"),
    [Input("set_slice", "value")],
)
def create_liver(slice_number):
    labels_folder_path = 'D:/data/train/kidney_1_dense/images/' + str(slice_number) + '.tif'  # Adjust the path accordingly
    all_img_types = []
    with rasterio.open(labels_folder_path) as image:
        all_img_types.append(image.read())
    img = all_img_types[0][0]
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@app.callback(
    Output("graph-liver-msk", "figure"),
    [Input("set_slice", "value")],
)
def create_liver_msk(slice_number):
    labels_folder_path = 'D:/data/train/kidney_1_dense/labels/' + str(slice_number) + '.tif'  # Adjust the path accordingly
    all_img_types = []
    with rasterio.open(labels_folder_path) as image:
        all_img_types.append(image.read())
    img = all_img_types[0][0]
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@app.callback(
    Output("graph-liver-msk2", "figure"),
    [Input("slice_slider", "value")],
)
def create_liver_msk2(slice_number):
    # Find better way to preload
    img = x[slice_number]
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
