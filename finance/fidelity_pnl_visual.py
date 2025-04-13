import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import base64
import io

# Create the Dash application
app = dash.Dash(__name__, title="Trading Activity Visualization")

# Define the layout with file upload component
app.layout = html.Div(
    [
        html.H1(
            "Trading Activity Analysis", style={"textAlign": "center", "margin": "20px"}
        ),
        html.Div(
            [
                html.H3(
                    "Upload Fidelity Trades Export (CSV):",
                    style={"marginBottom": "10px"},
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(
                        ["Drag and Drop or ", html.A("Select a CSV File")]
                    ),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px 0",
                    },
                    multiple=False,
                ),
                html.Div(
                    id="upload-status", style={"margin": "10px 0", "color": "green"}
                ),
            ],
            style={"margin": "20px"},
        ),
        html.Div(
            [
                html.Label("Select Timeframe:"),
                dcc.RadioItems(
                    id="timeframe-selector",
                    options=[
                        {"label": "All Time", "value": "all"},
                        {"label": "6 Months", "value": "6m"},
                        {"label": "3 Months", "value": "3m"},
                        {"label": "1 Month", "value": "1m"},
                    ],
                    value="all",
                    labelStyle={"display": "inline-block", "margin-right": "20px"},
                ),
            ],
            style={"margin": "20px"},
        ),
        html.Div(
            [
                html.Label("Show Trade Types:"),
                dcc.Checklist(
                    id="trade-type-filter",
                    options=[
                        {"label": "Buys", "value": "buy"},
                        {"label": "Sells", "value": "sell"},
                        {"label": "Shorts", "value": "short"},
                        {"label": "Covers", "value": "cover"},
                        {"label": "Options", "value": "option"},
                    ],
                    value=["buy", "sell", "short", "cover", "option"],
                    labelStyle={"display": "inline-block", "margin-right": "20px"},
                ),
            ],
            style={"margin": "20px"},
        ),
        dcc.Graph(id="trading-graph", style={"height": "600px"}),
        html.Div(
            [
                html.H2("Trading Statistics", style={"textAlign": "center"}),
                html.Div(
                    id="trading-stats",
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "justifyContent": "space-around",
                    },
                ),
            ],
            style={"margin": "30px"},
        ),
        # Store the processed data as an intermediate value
        dcc.Store(id="processed-data"),
    ]
)


def parse_uploaded_file(contents, filename):
    """Parse and validate the uploaded file"""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if "csv" in filename.lower():
            # Read the CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode("utf-8")), skipinitialspace=True
            )
            return process_fidelity_data(df), None
        else:
            return None, "Only CSV files are supported."
    except Exception as e:
        return None, f"Error processing file: {str(e)}"


def process_fidelity_data(df):
    """Process the uploaded Fidelity data"""
    # Clean up column names
    df.columns = [col.strip() for col in df.columns]

    # Check if this is a valid Fidelity export
    required_columns = [
        "Run Date",
        "Action",
        "Symbol",
        "Quantity",
        "Price ($)",
        "Amount ($)",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        cleaned_cols = [col.strip() for col in df.columns]
        df.columns = cleaned_cols
        # Try with cleaned column names
        missing_columns = [col for col in required_columns if col not in cleaned_cols]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Convert dates to datetime
    df["Run Date"] = pd.to_datetime(df["Run Date"], errors="coerce")

    # Sort by date
    df = df.sort_values("Run Date")

    # Determine trade types
    def get_trade_type(action):
        if not isinstance(action, str):
            return "unknown"

        action = action.upper()
        if "BOUGHT" in action and "SHORT COVER" not in action:
            return "buy"
        elif "SOLD" in action and "SHORT SALE" not in action:
            return "sell"
        elif "SHORT SALE" in action:
            return "short"
        elif "SHORT COVER" in action:
            return "cover"
        else:
            return "unknown"

    df["trade_type"] = df["Action"].apply(get_trade_type)

    # Determine if it's an option trade
    def is_option(row):
        if not isinstance(row["Symbol"], str) and not isinstance(row["Action"], str):
            return False

        symbol = str(row["Symbol"]).upper() if isinstance(row["Symbol"], str) else ""
        action = str(row["Action"]).upper() if isinstance(row["Action"], str) else ""

        return (
            "CALL" in symbol or "PUT" in symbol or "CALL" in action or "PUT" in action
        )

    df["is_option"] = df.apply(is_option, axis=1)

    # Filter out non-trade transactions
    trade_types = ["buy", "sell", "short", "cover", "unknown"]
    df = df[df["trade_type"].isin(trade_types)]

    # Calculate cumulative P&L
    # First, clean the Amount column to ensure it's numeric
    df["Amount ($)"] = pd.to_numeric(df["Amount ($)"], errors="coerce")

    # Calculate running sum of amounts (P&L)
    df["Cumulative P&L"] = df["Amount ($)"].cumsum()

    return df.to_dict("records")


def generate_trade_markers(df, selected_trade_types, include_options):
    """Generate scatter plot data for trade markers"""
    markers = []

    # Define colors and symbols for trade types
    trade_colors = {
        "buy": "green",
        "sell": "red",
        "short": "purple",
        "cover": "blue",
        "unknown": "gray",
    }

    trade_symbols = {
        "buy": "circle",
        "sell": "circle-open",
        "short": "triangle-down",
        "cover": "triangle-up",
        "unknown": "x",
    }

    # Filter trades based on selected types and options flag
    for trade_type in selected_trade_types:
        filtered_df = df[df["trade_type"] == trade_type]

        if not include_options and "option" not in selected_trade_types:
            filtered_df = filtered_df[~filtered_df["is_option"]]
        elif "option" in selected_trade_types and not include_options:
            filtered_df = filtered_df[filtered_df["is_option"]]

        if not filtered_df.empty:
            marker = go.Scatter(
                x=filtered_df["Run Date"],
                y=filtered_df["Cumulative P&L"],
                mode="markers",
                name=f"{trade_type.title()} Trades",
                marker=dict(
                    size=10,
                    color=trade_colors.get(trade_type, "gray"),
                    symbol=trade_symbols.get(trade_type, "circle"),
                ),
                hoverinfo="text",
                hovertext=[
                    f"Date: {row['Run Date'].strftime('%Y-%m-%d')}<br>"
                    f"Symbol: {row['Symbol']}<br>"
                    f"Action: {row['Action']}<br>"
                    f"Quantity: {row['Quantity']}<br>"
                    f"Amount: ${row['Amount ($)']:,.2f}<br>"
                    f"Net Value: ${row['Quantity'] * row['Price ($)']:,.2f}<br>"
                    f"P&L: ${row['Cumulative P&L']:,.2f}"
                    for _, row in filtered_df.iterrows()
                ],
            )
            markers.append(marker)

    return markers


@app.callback(
    Output("processed-data", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def store_data(contents, filename):
    """Process and store the uploaded file data"""
    if contents is None:
        return None, ""

    data, error = parse_uploaded_file(contents, filename)

    if error:
        return None, html.Div(error, style={"color": "red"})
    else:
        return data, html.Div(
            f"File '{filename}' uploaded successfully!", style={"color": "green"}
        )


@app.callback(
    [Output("trading-graph", "figure"), Output("trading-stats", "children")],
    [
        Input("timeframe-selector", "value"),
        Input("trade-type-filter", "value"),
        Input("processed-data", "data"),
    ],
    prevent_initial_call=True,
)
def update_graph(timeframe, selected_trade_types, data):
    """Update the graph based on filters and uploaded data"""
    if data is None:
        # Return empty figure and stats if no data
        fig = go.Figure()
        fig.update_layout(
            title="No data available. Please upload a Fidelity CSV export file.",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
        )
        return fig, []

    # Convert the dictionary data back to a DataFrame
    df = pd.DataFrame(data)

    # Convert dates back to datetime (as they were stored as strings in the dcc.Store)
    df["Run Date"] = pd.to_datetime(df["Run Date"])

    # Filter by timeframe
    if timeframe != "all" and not df.empty:
        end_date = df["Run Date"].max()
        if timeframe == "1m":
            start_date = end_date - pd.DateOffset(months=1)
        elif timeframe == "3m":
            start_date = end_date - pd.DateOffset(months=3)
        elif timeframe == "6m":
            start_date = end_date - pd.DateOffset(months=6)
        df = df[df["Run Date"] >= start_date]

    # Create the main figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if df.empty:
        fig.update_layout(
            title="No data available for the selected timeframe.",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L ($)",
        )
        return fig, []

    # Add the cumulative P&L line
    fig.add_trace(
        go.Scatter(
            x=df["Run Date"],
            y=df["Cumulative P&L"],
            mode="lines",
            name="Cumulative P&L",
            line=dict(color="blue", width=2),
        )
    )

    # Determine if options are included in the selection
    include_options = "option" in selected_trade_types
    # Remove 'option' from trade types since it's a flag, not a trade type
    trade_types = [t for t in selected_trade_types if t != "option"]

    # Add trade markers
    for trace in generate_trade_markers(df, trade_types, include_options):
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        title="Trading Activity and Cumulative P&L",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        template="plotly_white",
    )

    # Add a zero reference line
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")

    # Calculate stats for the trading summary
    total_trades = len(df)
    buy_trades = len(df[df["trade_type"] == "buy"])
    sell_trades = len(df[df["trade_type"] == "sell"])
    short_trades = len(df[df["trade_type"] == "short"])
    cover_trades = len(df[df["trade_type"] == "cover"])
    option_trades = len(df[df["is_option"]])
    trading_days = len(df["Run Date"].dt.date.unique())
    avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0

    # Net P&L for the period
    net_pl = df["Amount ($)"].sum()

    # Create the stats boxes
    stats_boxes = [
        html.Div(
            [html.H3("Total Trades"), html.P(f"{total_trades}")],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [html.H3("Buy Trades"), html.P(f"{buy_trades}", style={"color": "green"})],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [html.H3("Sell Trades"), html.P(f"{sell_trades}", style={"color": "red"})],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [
                html.H3("Options Trades"),
                html.P(f"{option_trades}", style={"color": "orange"}),
            ],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [
                html.H3("Short Sales"),
                html.P(f"{short_trades}", style={"color": "purple"}),
            ],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [
                html.H3("Short Covers"),
                html.P(f"{cover_trades}", style={"color": "blue"}),
            ],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [html.H3("Trading Days"), html.P(f"{trading_days}")],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        ),
        html.Div(
            [
                html.H3("Net P&L"),
                html.P(
                    f"${net_pl:,.2f}", style={"color": "green" if net_pl > 0 else "red"}
                ),
            ],
            style={
                "width": "200px",
                "padding": "20px",
                "margin": "10px",
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "textAlign": "center",
                "fontWeight": "bold",
            },
        ),
    ]

    return fig, stats_boxes


# Add a default view for when the app first loads
@app.callback(
    [
        Output("trading-graph", "figure", allow_duplicate=True),
        Output("trading-stats", "children", allow_duplicate=True),
    ],
    [Input("upload-data", "filename")],
    prevent_initial_call="initial_duplicate",
)
def initial_view(filename):
    if filename is not None:
        # If a file has been uploaded, don't override the view
        raise dash.exceptions.PreventUpdate

    # Create empty figure with instructions
    fig = go.Figure()
    fig.update_layout(
        title="Upload a Fidelity CSV export to visualize your trading activity",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        annotations=[
            dict(
                text="No data available. Please upload your Fidelity export file using the upload area above.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=16),
            )
        ],
    )

    # Create empty stats section
    empty_stats = html.Div(
        html.P(
            "Upload data to see trading statistics",
            style={"textAlign": "center", "color": "#888"},
        )
    )

    return fig, empty_stats


if __name__ == "__main__":
    app.run_server(debug=True)
