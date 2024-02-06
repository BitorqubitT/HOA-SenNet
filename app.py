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
# check how the input from a button is given
# Chek how they define a model then put it in a button and update it
# Maybe just wrap my code in a function and put that underneath a callback

# check populate_bigram_scatter for this
# line 724 -> perplexity>??????

# Add two buttons for the histogram

# Cleanup code, (format, remove useless stuff)

# Add other graphs (A way to view and compare mask, training slices?)
# Other graph shows score?

# remove these timers 

# ------------- I/O and data massaging ---------------------------------------------------

# Is there  a way to compute this on gpu?
labels_folder_path = 'D:/data/train/kidney_3_sparse/labels'  # Adjust the path accordingly
label_images = []
for file in sorted(os.listdir(labels_folder_path))[200:800]:
    if file.endswith(".tif"):
        label_image = tifffile.imread(os.path.join(labels_folder_path, file))
        label_image_downsampled = label_image[::1, ::1]
        label_images.append(torch.from_numpy(label_image_downsampled).float())

# Combine the first 100 images into one 3D tensor
all_images = torch.stack(label_images, dim=0)

# Extract coordinates of non-zero values (assuming binary segmentation)
non_zero_coords = all_images.nonzero()
med_img = non_zero_coords
med_img = all_images.numpy()

# Create mesh
# Step_size affects the level of detail and 200 is threshold. When volume over this threshold we graph it.
# Marching cube reduces the data complexity. 
# Without MC you dont have any surface roconstruction or isosurface extraction
verts, faces, _, _ = measure.marching_cubes(med_img, 200, step_size=3)
x, y, z = verts.T
i, j, k = faces.T
fig_mesh = go.Figure()
fig_mesh.add_trace(go.Mesh3d(x=z, y=y, z=x, opacity=0.2, i=k, j=j, k=i))

def path_to_coords(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point"""
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.array(indices_str, dtype=float)


def largest_connected_component(mask):
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (np.argmax(sizes) + 1)


# ------------- Define App Layout ---------------------------------------------------

mesh_card = dbc.Card(
    [
        dbc.CardHeader("3D mesh representation of the image data and annotation"),
        dbc.CardBody([dcc.Graph(id="graph-helper", figure=fig_mesh)]),
    ]
)

# Buttons
# Can I remove these buttons?
button_gh = dbc.Button(
    "Learn more",
    id="howto-open",
    outline=True,
    color="secondary",
    # Turn off lowercase transformation for class .button in stylesheet
    style={"textTransform": "none"},
)

button_howto = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-covid-xray",
    id="gh-link",
    style={"text-transform": "none"},
)


# New button which holds params that can alter the processing of the image
# Check other app, might need to set col + html title 
button_stepsize = dcc.Dropdown(
    id = "set_stepsize",
    options = [{"label": str(i), "value": i}
               for i in range(3, 7)
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
    [Output("graph-histogram", "figure"), Output("roi-warning", "is_open")],
    [Input("annotations", "data")],
)
def update_histo(annotations):
    if (
        annotations is None
        or annotations.get("x") is None
        or annotations.get("z") is None
    ):
        return dash.no_update, dash.no_update
    # Horizontal mask for the xy plane (z-axis)
    path = path_to_coords(annotations["z"]["path"])
    rr, cc = draw.polygon(path[:, 1] / spacing[1], path[:, 0] / spacing[2])
    if len(rr) == 0 or len(cc) == 0:
        return dash.no_update, dash.no_update
    mask = np.zeros(img.shape[1:])
    mask[rr, cc] = 1
    mask = ndimage.binary_fill_holes(mask)
    # top and bottom, the top is a lower number than the bottom because y values
    # increase moving down the figure
    top, bottom = sorted([int(annotations["x"][c] / spacing[0]) for c in ["y0", "y1"]])
    intensities = med_img[top:bottom, mask].ravel()
    if len(intensities) == 0:
        return dash.no_update, dash.no_update
    hi = exposure.histogram(intensities)
    fig = px.bar(
        x=hi[1],
        y=hi[0],
        # Histogram
        labels={"x": "intensity", "y": "count"},
    )
    fig.update_layout(dragmode="select", title_font=dict(size=20, color="blue"))
    return fig, False


@app.callback(
    Output("annotations", "data"),
    [State("annotations", "data")],
)

def update_annotations(relayout1, relayout2, annotations):
    if relayout1 is not None and "shapes" in relayout1:
        if len(relayout1["shapes"]) >= 1:
            shape = relayout1["shapes"][-1]
            annotations["z"] = shape
        else:
            annotations.pop("z", None)
    elif relayout1 is not None and "shapes[2].path" in relayout1:
        annotations["z"]["path"] = relayout1["shapes[2].path"]

    if relayout2 is not None and "shapes" in relayout2:
        if len(relayout2["shapes"]) >= 1:
            shape = relayout2["shapes"][-1]
            annotations["x"] = shape
        else:
            annotations.pop("x", None)
    elif relayout2 is not None and (
        "shapes[2].y0" in relayout2 or "shapes[2].y1" in relayout2
    ):
        annotations["x"]["y0"] = relayout2["shapes[2].y0"]
        annotations["x"]["y1"] = relayout2["shapes[2].y1"]
    return annotations

app.clientside_callback(
    """
function(surf, fig){
        let fig_ = {...fig};
        fig_.data[1] = surf;
        return fig_;
    }
""",
    output=Output("graph-helper", "figure"),
    inputs=[Input("occlusion-surface", "data"),],
    state=[State("graph-helper", "figure"),],
)

@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_props_check=False)
