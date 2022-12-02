import numpy as np
import plotly.graph_objects as go


def volume_3d(blob, title, opacity=0.1, surface_count=15, colorscale="PuBu"):
    x, y, z = blob.shape
    max_dim = max(x, y, z)
    x_pad = (max_dim - x) / 2
    y_pad = (max_dim - y) / 2
    z_pad = (max_dim - z) / 2

    data = np.pad(
        blob,
        (
            (int(np.ceil(x_pad)), int(np.floor(x_pad))),
            (int(np.ceil(y_pad)), int(np.floor(y_pad))),
            (int(np.ceil(z_pad)), int(np.floor(z_pad))),
        ),
        "constant",
        constant_values=0,
    )

    X, Y, Z = np.mgrid[
      -1: 1: max_dim * 1j, -1: 1: max_dim * 1j, -1: 1: max_dim * 1j
      ]

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data.flatten(),
            isomin=float(blob[blob > 0].min()),
            isomax=float(blob[blob > 0].max()),
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale
        )
    )

    fig.update_layout(
        title={
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        height=600
    )

    return fig
