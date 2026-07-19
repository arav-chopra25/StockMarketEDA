"""Plotly-based visualization helpers for stock market analytics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _resolve_date_column(dataframe: pd.DataFrame, requested_column: str) -> str:
    if requested_column in dataframe.columns:
        return requested_column
    if requested_column == "date" and "timestamp" in dataframe.columns:
        return "timestamp"
    if requested_column == "timestamp" and "date" in dataframe.columns:
        return "date"
    raise KeyError(f"Column '{requested_column}' not found in dataframe.")


def _base_layout(fig: go.Figure, title: str, xaxis_title: str = "Date", yaxis_title: str = "Value") -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title=xaxis_title, rangeslider_visible=True)
    fig.update_yaxes(title=yaxis_title)
    return fig


def line_chart(dataframe: pd.DataFrame, x_col: str = "date", y_col: str = "close", title: str = "Price Trend") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[y_col], mode="lines", name=y_col))
    return _base_layout(fig, title, xaxis_title=x_col, yaxis_title=y_col)


def candlestick_chart(dataframe: pd.DataFrame, x_col: str = "date") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=dataframe[x_col],
                open=dataframe["open"],
                high=dataframe["high"],
                low=dataframe["low"],
                close=dataframe["close"],
                name="Candlestick",
            )
        ]
    )
    return _base_layout(fig, "Candlestick Chart", xaxis_title=x_col, yaxis_title="Price")


def ohlc_chart(dataframe: pd.DataFrame, x_col: str = "date") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure(
        data=[
            go.Ohlc(
                x=dataframe[x_col],
                open=dataframe["open"],
                high=dataframe["high"],
                low=dataframe["low"],
                close=dataframe["close"],
                name="OHLC",
            )
        ]
    )
    return _base_layout(fig, "OHLC Chart", xaxis_title=x_col, yaxis_title="Price")


def area_chart(dataframe: pd.DataFrame, x_col: str = "date", y_col: str = "close", title: str = "Area Chart") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dataframe[x_col],
            y=dataframe[y_col],
            mode="lines",
            fill="tozeroy",
            name=y_col,
        )
    )
    return _base_layout(fig, title, xaxis_title=x_col, yaxis_title=y_col)


def scatter_plot(dataframe: pd.DataFrame, x_column: str, y_column: str, title: str = "Scatter Plot") -> go.Figure:
    fig = px.scatter(dataframe, x=x_column, y=y_column, title=title)
    return _base_layout(fig, title, xaxis_title=x_column, yaxis_title=y_column)


def bubble_chart(dataframe: pd.DataFrame, x_column: str, y_column: str, size_column: str, title: str = "Bubble Chart") -> go.Figure:
    fig = px.scatter(dataframe, x=x_column, y=y_column, size=size_column, title=title, size_max=40)
    return _base_layout(fig, title, xaxis_title=x_column, yaxis_title=y_column)


def histogram(dataframe: pd.DataFrame, column: str, title: str = "Histogram") -> go.Figure:
    fig = px.histogram(dataframe, x=column, nbins=40, title=title)
    return _base_layout(fig, title, xaxis_title=column, yaxis_title="Count")


def box_plot(dataframe: pd.DataFrame, column: str, title: str = "Box Plot") -> go.Figure:
    fig = px.box(dataframe, y=column, title=title)
    return _base_layout(fig, title, xaxis_title="", yaxis_title=column)


def violin_plot(dataframe: pd.DataFrame, column: str, title: str = "Violin Plot") -> go.Figure:
    fig = px.violin(dataframe, y=column, box=True, points="all", title=title)
    return _base_layout(fig, title, xaxis_title="", yaxis_title=column)


def correlation_heatmap(correlation_frame: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    fig = px.imshow(correlation_frame, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title=title)
    return _base_layout(fig, title, xaxis_title="Features", yaxis_title="Features")


def treemap(dataframe: pd.DataFrame, path_columns: list[str], value_column: str, title: str = "Treemap") -> go.Figure:
    fig = px.treemap(dataframe, path=path_columns, values=value_column, title=title)
    return _base_layout(fig, title, xaxis_title="", yaxis_title=value_column)


def sunburst_chart(dataframe: pd.DataFrame, path_columns: list[str], value_column: str, title: str = "Sunburst") -> go.Figure:
    fig = px.sunburst(dataframe, path=path_columns, values=value_column, title=title)
    return _base_layout(fig, title, xaxis_title="", yaxis_title=value_column)


def pie_chart(dataframe: pd.DataFrame, names_column: str, values_column: str, title: str = "Pie Chart") -> go.Figure:
    fig = px.pie(dataframe, names=names_column, values=values_column, title=title)
    return _base_layout(fig, title, xaxis_title="", yaxis_title=values_column)


def bar_chart(dataframe: pd.DataFrame, x_column: str, y_column: str, title: str = "Bar Chart") -> go.Figure:
    fig = px.bar(dataframe, x=x_column, y=y_column, title=title)
    return _base_layout(fig, title, xaxis_title=x_column, yaxis_title=y_column)


def moving_average_chart(dataframe: pd.DataFrame, x_col: str = "date", y_col: str = "close", ma_columns: list[str] | None = None) -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    ma_columns = ma_columns or []
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[y_col], mode="lines", name=y_col))
    for column in ma_columns:
        if column in dataframe.columns:
            fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[column], mode="lines", name=column))
    return _base_layout(fig, "Moving Average Chart", xaxis_title=x_col, yaxis_title="Price")


def ema_chart(dataframe: pd.DataFrame, x_col: str = "date", y_col: str = "close", ema_columns: list[str] | None = None) -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    ema_columns = ema_columns or []
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[y_col], mode="lines", name=y_col))
    for column in ema_columns:
        if column in dataframe.columns:
            fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[column], mode="lines", name=column))
    return _base_layout(fig, "EMA Chart", xaxis_title=x_col, yaxis_title="Price")


def rsi_chart(dataframe: pd.DataFrame, x_col: str = "date", rsi_column: str = "rsi") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[rsi_column], mode="lines", name="RSI"))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    return _base_layout(fig, "RSI Chart", xaxis_title=x_col, yaxis_title="RSI")


def macd_chart(dataframe: pd.DataFrame, x_col: str = "date", macd_column: str = "macd", signal_column: str = "signal_line") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[macd_column], mode="lines", name="MACD"))
    if signal_column in dataframe.columns:
        fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[signal_column], mode="lines", name="Signal"))
    if "macd_histogram" in dataframe.columns:
        fig.add_trace(go.Bar(x=dataframe[x_col], y=dataframe["macd_histogram"], name="Histogram"))
    return _base_layout(fig, "MACD Chart", xaxis_title=x_col, yaxis_title="MACD")


def volume_chart(dataframe: pd.DataFrame, x_col: str = "date", volume_column: str = "volume") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dataframe[x_col], y=dataframe[volume_column], name=volume_column))
    return _base_layout(fig, "Volume Chart", xaxis_title=x_col, yaxis_title=volume_column)


def daily_return_chart(dataframe: pd.DataFrame, x_col: str = "date", return_column: str = "daily_return") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[return_column], mode="lines", name=return_column))
    return _base_layout(fig, "Daily Return Chart", xaxis_title=x_col, yaxis_title=return_column)


def rolling_mean_chart(dataframe: pd.DataFrame, x_col: str = "date", y_col: str = "close", rolling_column: str = "rolling_mean_20") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[y_col], mode="lines", name=y_col))
    if rolling_column in dataframe.columns:
        fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[rolling_column], mode="lines", name=rolling_column))
    return _base_layout(fig, "Rolling Mean Chart", xaxis_title=x_col, yaxis_title=y_col)


def volatility_chart(dataframe: pd.DataFrame, x_col: str = "date", volatility_column: str = "volatility_20") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataframe[x_col], y=dataframe[volatility_column], mode="lines", name=volatility_column))
    return _base_layout(fig, "Volatility Chart", xaxis_title=x_col, yaxis_title=volatility_column)


def animated_time_series(dataframe: pd.DataFrame, x_col: str = "date", y_col: str = "close", animation_frame: str = "year") -> go.Figure:
    x_col = _resolve_date_column(dataframe, x_col)
    fig = px.line(dataframe, x=x_col, y=y_col, animation_frame=animation_frame, title="Animated Time Series")
    return _base_layout(fig, "Animated Time Series", xaxis_title=x_col, yaxis_title=y_col)
