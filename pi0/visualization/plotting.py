import numpy as np
import pandas as pd
import yaml
import sys, os
import importlib

# We will use Plotly for plotting 3D event displays
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot

from mlreco.visualization import voxels

def plot_event(coords, label, points=None):
    """
    This removes ghost points. 
    """
    trace = voxels.scatter_label(coords, 
                                 label)
    if points is not None:
        trace_primaries = go.Scatter3d(x=points[:, 0],
                               y=points[:, 1],
                               z=points[:, 2],
                               mode='markers',
                               marker=dict(
                                   color='red',
                                   size=4,
                                   opacity=0.7,
                                   symbol='diamond'
                                   ),
                               name='Points'
                               )
        trace.append(trace_primaries)
    plot_labels = {
        'data': trace,
        'layout': go.Layout(
            margin={'l': 20, 'b': 20,
                    't': 20, 'r': 20},
            plot_bgcolor="#fff",
            paper_bgcolor="#fff"
        )
    }
    return plot_labels
