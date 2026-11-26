"""
Academic Plotly visualisations for the Learned Index Visualizer.

Palette & Aesthetics:
- Off-white paper background (#F7F8FA) and crisp white plot area (#FFFFFF)
- Subtle gridlines (#E2E8F0)
- Primary Blue (#315E91): Binary Search & Actual Data Points
- Muted Teal (#368A83): Learned Index & Linear Regression Model
- Slate Gray (#64748B): Baseline References & Secondary traces
- Amber (#D97706): Naive Learned Index & Error limits
"""

import numpy as np
import plotly.graph_objects as go


# Theme palette
PALETTE = {
    "primary": "#315E91",       # Navy/Steel Blue
    "teal": "#368A83",          # Muted Teal
    "amber": "#D97706",         # Amber/Ochre
    "crimson": "#DC2626",       # Alert Red
    "slate": "#64748B",         # Slate
    "charcoal": "#202936",      # Text
    "grid": "#E2E8F0",          # Subtle grid
    "plot_bg": "#FFFFFF",       # Plot canvas
    "paper_bg": "#F7F8FA",      # Paper background
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=PALETTE["paper_bg"],
    plot_bgcolor=PALETTE["plot_bg"],
    font=dict(color=PALETTE["charcoal"], family="Inter, -apple-system, sans-serif", size=11),
    margin=dict(l=50, r=30, t=40, b=45),
    xaxis=dict(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["grid"],
        showline=True,
        linecolor=PALETTE["grid"],
        ticks="outside",
    ),
    yaxis=dict(
        gridcolor=PALETTE["grid"],
        zerolinecolor=PALETTE["grid"],
        showline=True,
        linecolor=PALETTE["grid"],
        ticks="outside",
    ),
)


def _apply_theme(fig, height=380):
    fig.update_layout(**LAYOUT_DEFAULTS, height=height)
    return fig


def regression_plot(keys, actual_pos, predicted_pos, max_error=None, title="Key Value vs. Sorted Index Position"):
    """Plot actual data points, linear regression prediction line, and shaded error bounds."""
    fig = go.Figure()

    # Shaded error band
    if max_error is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([keys, keys[::-1]]),
            y=np.concatenate([predicted_pos - max_error, (predicted_pos + max_error)[::-1]]),
            fill="toself",
            fillcolor="rgba(54, 138, 131, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"Search Bound (±{max_error})",
            showlegend=True,
        ))

    # Actual index positions
    fig.add_trace(go.Scatter(
        x=keys, y=actual_pos, mode="markers",
        marker=dict(color=PALETTE["primary"], size=3, opacity=0.4),
        name="Actual Position",
    ))

    # Learned regression model prediction line
    fig.add_trace(go.Scatter(
        x=keys, y=predicted_pos, mode="lines",
        line=dict(color=PALETTE["teal"], width=2),
        name="Learned Model (y = mx + c)",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=PALETTE["charcoal"])),
        xaxis_title="Search Key Value",
        yaxis_title="Sorted Index Position",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _apply_theme(fig, height=420)


def key_distribution_plot(data, n_bins=75, title="Key Distribution Histogram"):
    """Histogram showing key density and distribution."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data, nbinsx=n_bins,
        marker_color=PALETTE["primary"],
        opacity=0.75,
        name="Key Frequency",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=PALETTE["charcoal"])),
        xaxis_title="Key Value",
        yaxis_title="Frequency Count",
        showlegend=False,
    )
    return _apply_theme(fig, height=330)


def error_distribution_plot(errors, title="Prediction Error (Residuals) Distribution"):
    """Histogram of prediction residuals."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=60,
        marker_color=PALETTE["teal"],
        opacity=0.8,
        name="Absolute Error",
    ))
    mean_val = float(np.mean(errors))
    fig.add_vline(x=mean_val, line_dash="dash", line_color=PALETTE["charcoal"],
                  annotation_text=f"Mean: {mean_val:.1f}", annotation_position="top right")

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=PALETTE["charcoal"])),
        xaxis_title="Absolute Error |Predicted - Actual| (Positions)",
        yaxis_title="Count",
        showlegend=False,
    )
    return _apply_theme(fig, height=340)


def error_vs_key_plot(keys, errors, max_error=None, title="Prediction Error vs. Key Value"):
    """Scatter plot of signed prediction errors across the key space."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=keys, y=errors, mode="markers",
        marker=dict(color=PALETTE["slate"], size=3, opacity=0.45),
        name="Residual (ŷ - y)",
    ))
    fig.add_hline(y=0, line_dash="solid", line_color=PALETTE["charcoal"], line_width=1)
    if max_error:
        fig.add_hline(y=max_error, line_dash="dot", line_color=PALETTE["amber"],
                      annotation_text=f"+Max ({max_error})", annotation_position="top right")
        fig.add_hline(y=-max_error, line_dash="dot", line_color=PALETTE["amber"],
                      annotation_text=f"-Max (-{max_error})", annotation_position="bottom right")

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=PALETTE["charcoal"])),
        xaxis_title="Key Value",
        yaxis_title="Prediction Error (Positions)",
        showlegend=False,
    )
    return _apply_theme(fig, height=340)


def benchmark_comparison_bar(bs_time, naive_time, opt_time, unit="μs", title="Average Lookup Latency"):
    """Bar chart comparing Binary Search, Naive Learned Index, and Optimized Learned Index."""
    methods = ["Binary Search", "Naive Learned Index", "Optimized Learned Index"]
    times = [bs_time, naive_time, opt_time]
    colors = [PALETTE["primary"], PALETTE["amber"], PALETTE["teal"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=methods, y=times,
        marker_color=colors,
        text=[f"{t:.2f} {unit}" for t in times],
        textposition="outside",
        textfont=dict(size=12, color=PALETTE["charcoal"]),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=PALETTE["charcoal"])),
        yaxis_title=f"Time per Query ({unit})",
        yaxis=dict(range=[0, max(times) * 1.25]),
        showlegend=False,
    )
    return _apply_theme(fig, height=380)


def search_window_comparison_plot(predicted_pos, lo, hi, actual_pos, data_size):
    """Visualizes the local search window relative to the entire index address space."""
    fig = go.Figure()

    # Full array span
    fig.add_trace(go.Bar(
        x=[data_size], y=["Index Space"], orientation="h",
        marker_color="#E2E8F0", name="Full Table Index", showlegend=True,
    ))

    # Error bounded window
    fig.add_shape(
        type="rect", x0=lo, x1=hi, y0=-0.35, y1=0.35,
        fillcolor="rgba(54, 138, 131, 0.25)",
        line=dict(color=PALETTE["teal"], width=1.5),
    )

    # Predicted position marker
    fig.add_trace(go.Scatter(
        x=[predicted_pos], y=["Index Space"], mode="markers",
        marker=dict(color=PALETTE["amber"], size=12, symbol="triangle-down"),
        name=f"Predicted Pos: {predicted_pos:,}",
    ))

    # Actual position marker if present
    if actual_pos is not None and actual_pos >= 0:
        fig.add_trace(go.Scatter(
            x=[actual_pos], y=["Index Space"], mode="markers",
            marker=dict(color=PALETTE["primary"], size=12, symbol="line-ns", line=dict(width=3, color=PALETTE["primary"])),
            name=f"Actual Pos: {actual_pos:,}",
        ))

    fig.update_layout(
        title=dict(text=f"Search Window: [{lo:,} ... {hi:,}] ({hi - lo + 1:,} elements / {((hi-lo+1)/data_size)*100:.2f}% of table)",
                   font=dict(size=12, color=PALETTE["charcoal"])),
        xaxis_title="Index Position",
        height=180,
        xaxis=dict(range=[-data_size * 0.02, data_size * 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
    )
    return _apply_theme(fig, height=190)


def latency_distribution_box(bs_times, naive_times, opt_times):
    """Box plots of per-query latency distributions."""
    fig = go.Figure()
    for name, times, col in [
        ("Binary Search", bs_times, PALETTE["primary"]),
        ("Naive Learned", naive_times, PALETTE["amber"]),
        ("Optimized Learned", opt_times, PALETTE["teal"]),
    ]:
        fig.add_trace(go.Box(
            y=times, name=name, marker_color=col, boxmean=True,
        ))

    fig.update_layout(
        title=dict(text="Per-Query Lookup Latency Distribution", font=dict(size=13, color=PALETTE["charcoal"])),
        yaxis_title="Latency (μs)",
        showlegend=False,
    )
    return _apply_theme(fig, height=380)
