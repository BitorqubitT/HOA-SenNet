from time import time

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
import tifffile
import torch


external_stylesheets = [dbc.themes.BOOTSTRAP, '/assets/style.css']
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)
server = app.server

#set param for image size
WIDTH = 500
HEIGHT = 500

# TODO:

#check if i can convert the type to something which uses less memory. (flt? int16?)
#https://github.com/plotly/dash-slicer/blob/main/dash_slicer/slicer.py
# read performance tips (chat?).
# How much do I want to load in memory

# dcc? store?

# Set good params

# Add other graphs (A way to view and compare mask, training slices?)

# Show score

# ------------- Define App Layout ---------------------------------------------------

dev_tools_props_check=False
update_title=None

mesh_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("3D mesh representation of the image data and annotation"),
        dcc.Graph(id="graph-helper"),
        ]
    )
    ]
)

slice_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("2D slice of kidney"),
        dcc.Graph(id="graph-kidney"),
        ]
    )
    ]
)

mask_truth_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("2D slice of kidney"),
        dcc.Graph(id="graph-kidney-msk-truth"),
        ]
    )
    ]
)

bar_scores = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("Overview scores"),
        dcc.Graph(id="scores_overview"),
        ]
    )
    ]
)

# check effect of resizing
# Check if I can implement this similar to the rest
def pre_load_image():
    labels_folder_path = 'D:/data/train/kidney_1_dense/images/'  # Adjust the path accordingly
    label_images = []
    for file in sorted(os.listdir(labels_folder_path))[800:950]:
        if file.endswith(".tif"):
            label_image = tifffile.imread(os.path.join(labels_folder_path, file))
            label_image = label_image[0::1, 0::1]
            label_images.append(label_image)
    # Combine the first 100 images into one 3D tensor
    x = np.array(label_images)
    return x

def pre_load_label():
    labels_folder_path = 'D:/data/train/kidney_1_dense/labels/'  # Adjust the path accordingly
    label_images = []
    for file in sorted(os.listdir(labels_folder_path))[800:950]:
        if file.endswith(".tif"):
            label_image = tifffile.imread(os.path.join(labels_folder_path, file))
            label_image = label_image[0::1, 0::1]
            label_images.append(label_image)
    # Combine the first 100 images into one 3D tensor
    x = np.array(label_images)
    print(" imagesize", x.shape)
    return x

# Get this from the other matrices?
"""
def get_diceloss(slice):
    return helper.dice_loss(xx[slice], yy[slice])
"""


x = pre_load_image()
xz = pre_load_label()

# New button which holds params that can alter the processing of the image
# Check other app, might need to set col + html title 
button_stepsize = dcc.Dropdown(
    id = "set_stepsize",
    options = [{"label": str(i), "value": i}
               for i in range(1, 10)
              ],
    value=3,
    className="Paramdrop"
)

button_threshold = dcc.Dropdown(
    id = "set_threshold",
    options = [{"label": str(i), "value": i}
               for i in range(100, 300, 100)
              ],
    value=200,
    className="Paramdrop"
)

slider_kidney = dcc.Slider(
    1, 300, 10,
    id = "slice_slider",
    value = 200,
    marks = None,
    className="Sliderboy"
    )

# create two buttons for liver pic
# We can put a body in here, the define the body above which holds: buttons and graph
# might want to use storesssssssssssssss
# https://dash.plotly.com/dash-core-components/loading


app.layout = html.Div(
    [
        dbc.Container(
            [
            dbc.Row([html.H2("Blood vessel segmentation")]),

            dbc.Row([
                dbc.Col(mesh_card, width=6),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(html.P(["Choose a step_size:"]), width=3),
                        dbc.Col(button_stepsize, width=4)
                    ]),
                    dbc.Row([
                        dbc.Col(html.P(["Choose a threshold:"]), width=3),
                        dbc.Col(button_threshold, width=4)
                    ]),
                    dbc.Row([
                        dbc.Col(html.P(["Choose a slice height:"]), width=3),
                        dbc.Col(slider_kidney, width=4)
                    ])
                ])
            ], className="row-spacing"),

            dbc.Row([
                dbc.Col(slice_card, width = 4),
                dbc.Col(mask_truth_card, width = 4)
                ]),

            dbc.Row([dbc.Col(bar_scores, width = 9)
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
    # What happens when i put this in memory so we dont have to compute everytime?
    # We actually only have to recopmute when we touch step_size etc.
    # Is this what we are doing now or not?
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
    fig = go.Figure(layout={})
    fig.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))

    # add plane
    # Assuming you have the dimensions of your volume
    volume_shape = all_images.shape

    # Define the height of the horizontal plane
    print(volume_shape[0])
    plane_height = volume_shape[0] // 2  # You can adjust this as needed
    print("slider value is:", slice_slider)
    print("height is:", plane_height)
    # Create grid for the plane
    x_plane, y_plane = np.meshgrid(np.arange(volume_shape[2]), np.arange(volume_shape[1]))
    z_plane = np.full_like(x_plane, plane_height)

    fig.add_trace(go.Surface(z=z_plane, y=y_plane, x=x_plane, opacity=0.5, showscale=False))
    return fig

# Can clean up this code by having 2 outputs in the same callback
# Dont need two functions then
# Cleaner way to share the input

# Check what happens if we load the whole image in one go.
# then only change the slice 

@app.callback(
    Output("graph-kidney", "figure"),
    [Input("slice_slider", "value")],
)
def create_liver_msk2(slice_number):
    # Find better way to preload
    img = x[slice_number]
    fig = px.imshow(img, width=450, height=750)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@app.callback(
    Output("graph-kidney-msk-truth", "figure"),
    [Input("slice_slider", "value")],
)
def create_liver_msk2(slice_number):
    # Find better way to preload
    img = xz[slice_number]
    fig = px.imshow(img, width=450, height=750)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@app.callback(
    Output("scores_overview", "figure"),
    [Input("slice_slider", "value")]
)
def scores(value):
    x = ['A', 'B', 'C', 'D']
    y = [3, 5, 7, 9]
    print(" we hawtadjaopiwnjd")
    fig = go.Figure(layout={})
    fig.add_bar(x=x, y=y)
    return fig





if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
