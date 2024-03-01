import numpy as np
from skimage import measure
import os
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import tifffile
import torch
import pandas as pd

external_stylesheets = [dbc.themes.BOOTSTRAP, '/assets/style.css']
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)
server = app.server

# TODO:
# check convert the type to something which uses less memory. (int16?)

# Create cards which hold graphs etc.

mesh_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("3D representation of blood vessels in kidney"),
        dcc.Graph(id="graph-helper"),
        ]
    )
    ]
)

slice_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("2D slice of kidney"),
        dcc.Graph(id="graph-kidney", className="test"),
        ],
    className="pic_card"
    )
    ]
)

mask_truth_card = dbc.Card(
    [ dcc.Loading(
        children = [
        dbc.CardHeader("2D segmented vessels"),
        dcc.Graph(id="graph-kidney-msk-truth", className="test"),
        ],
    className="pic_card"
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

# Pre load functions
def pre_load_image():
    labels_folder_path = './data/train/kidney_3_sparse/images/' 
    label_images = []
    for file in sorted(os.listdir(labels_folder_path))[0:950]:
        if file.endswith(".tif"):
            label_image = tifffile.imread(os.path.join(labels_folder_path, file))
            label_image = label_image[0::4, 0::4]
            label_images.append(label_image)

    x = np.array(label_images)
    return x

def pre_load_label():
    labels_folder_path = './data/results/'
    label_images = []
    for file in sorted(os.listdir(labels_folder_path))[0:950]:
        if file.endswith(".tif"):
            label_image = tifffile.imread(os.path.join(labels_folder_path, file))
            label_image = label_image[0::4, 0::4]
            label_images.append(label_image)

    x = np.array(label_images)
    return x

def pre_load_slice_scores():
    df = pd.read_csv("scores_per_slice.csv")
    return df["scores"]

def pre_load_masked_images():
    # create function maybe use dcc loader
    # Can combine this function with pre_load_label
    labels_folder_path = './data/train/kidney_3_sparse/labels'
    label_images = []
    for file in sorted(os.listdir(labels_folder_path))[0:950]:
        if file.endswith(".tif"):
            label_image = tifffile.imread(os.path.join(labels_folder_path, file))
            label_image_downsampled = label_image[::1, ::1]
            label_images.append(torch.from_numpy(label_image_downsampled).float())

    all_images = torch.stack(label_images, dim=0)

    return all_images.numpy()

# load data, so we only do it once.
med_img = pre_load_masked_images()
x = pre_load_image()
labels = pre_load_label()
scores = pre_load_slice_scores()
colors = ["blue"] * len(scores)

# create all buttons
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
    1, 1000, 10,
    id = "slice_slider",
    value = 200,
    marks = None,
    className="Sliderboy"
    )

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
                ], className="row-spacing"),
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
    # Step_size affects the level of detail and 200 is threshold. When volume over this threshold we graph it.
    # Without MC you dont have any surface reconstruction or isosurface extraction
    verts, faces, _, _ = measure.marching_cubes(med_img, threshold, step_size=step_size)
    x, y, z = verts.T
    i, j, k = faces.T
    fig = go.Figure(layout={})
    fig.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))

    # Create grid for the plane
    volume_shape = med_img.shape
    x_plane, y_plane = np.meshgrid(np.arange(volume_shape[2]), np.arange(volume_shape[1]))
    z_plane = np.full_like(x_plane, slice_slider)

    fig.add_trace(go.Surface(z=z_plane, y=y_plane, x=x_plane, opacity=0.5, showscale=False))
    fig.update_layout(
        margin={"t": 0, "b": 0, "r": 0, "l": 0, "pad": 0},
    )
    return fig

@app.callback(
    Output("graph-kidney", "figure"),
    [Input("slice_slider", "value")],
)
def create_liver_img(slice_number):
    img = x[slice_number]
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(margin={"t": 10, "b": 10, "r": 0, "l": 0, "pad": 0},)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@app.callback(
    Output("graph-kidney-msk-truth", "figure"),
    [Input("slice_slider", "value")],
)
def create_liver_msk2(slice_number):
    img = labels[slice_number]
    fig = px.imshow(img)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(margin={"t": 10, "b": 10, "r": 0, "l": 0, "pad": 0},)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

@app.callback(
    Output("scores_overview", "figure"),
    [Input("slice_slider", "value")]
)
def scores(value):
    x = [i for i in range(len(scores))]
    colors[value] = "red"

    fig = go.Figure(layout={})
    fig.add_bar(x=x, y=scores, marker=dict(color=colors))
    fig.update_layout(
        margin={"t": 5, "b": 5, "r": 5, "l": 5, "pad": 0},
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
