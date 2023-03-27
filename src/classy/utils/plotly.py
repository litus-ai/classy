import numpy as np

from .optional_deps import requires

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


@requires("plotly.graph_objects")
def boxplot(y: np.ndarray, x_name: str, y_name: str, color: str):
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=y,
            name=x_name,
            marker_color=color,
            boxmean="sd",
        ),
    )
    fig.update_layout(
        yaxis_title=y_name,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig
