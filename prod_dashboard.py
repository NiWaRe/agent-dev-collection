import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import weave
from datetime import datetime, timedelta
from typing import List, Optional
from pydantic import BaseModel, Field
import itertools
import json

# TODO: 
# - proof-read the whole project (especially the segments)
# - think about the find_model_name_recursive (commit first version without this)
# - turn this into a cookbook and commit - then share all the new cookbooks in the channel
# - add costs to the dashboard and also the possibility to aggregate across multiple projects
# - add clear list of requirements that the product design team can pick up together with Scott and eng
# - would be nice to be able to only select a specific trace ID (only prod calls)
# - check out new custom cost and feedback query functions 

# Define available projects
AVAILABLE_PROJECTS = [
    "wandb-smle/weave-cookboook-demo",
    "wandb-smle/weave-rag-experiments",
    "griffin_wb/prod-evals-aug",
    # Add more projects as needed
]

# Define model names to search for
MODEL_NAMES = [
    "gemini/gemini-1.5-flash",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "claude-3-haiku-20240307",
    "gpt-4o"
    # Add more models as needed
]

# Enhance the dashboard appearance
st.set_page_config(layout="wide", page_title="Weave LLM Monitoring Dashboard")
st.markdown("""
    <style>
    .main {
        padding-left: 50px;
        padding-right: 50px;
    }
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Weave client
@st.cache_resource
def init_weave_client(project_name: str):
    try:
        client = weave.init(project_name)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Weave client for project '{project_name}': {e}")
        return None

# Function to fetch calls data from Weave with a limit to prevent hanging
def fetch_calls(client, project_id: str, start_time: datetime, end_time: datetime, trace_roots_only: bool, limit: int = 1000):
    filter_params = {
        "project_id": project_id,
        "filter": {
            "started_at": {
                "$gte": start_time.isoformat(),
                "$lt": end_time.isoformat()
            },
            "trace_roots_only": trace_roots_only
        },
        "expand_columns": ["inputs.example", "inputs.model"],
        "sort_by": [{"field": "started_at", "direction": "desc"}],
        "include_costs": True,
        "include_feedback": True,
    }
    try:
        calls_stream = client.server.calls_query_stream(filter_params)
        # Limit the number of calls fetched to prevent infinite loops
        calls = list(itertools.islice(calls_stream, limit))
        st.write(f"Fetched {len(calls)} calls.")
        return calls
    except Exception as e:
        st.error(f"Error fetching calls: {e}")
        return []

# TODO: issues with this approach
# - this will have to find for eval objects: call.inputs["model"].chat_model.chat_model 
# - this approach doesn't work with traces that have multiple different models as children (like for evals! completely different appraoch neede!)
# - the integrations can't be tracked - e.g. https://wandb.ai/wandb-smle/weave-cookboook-demo/weave/traces?filters=%7B%22items%22%3A%5B%7B%22id%22%3A0%2C%22field%22%3A%22inputs.prompt%22%2C%22operator%22%3A%22%28string%29%3A+contains%22%2C%22value%22%3A%22Still%22%7D%5D%2C%22logicOperator%22%3A%22and%22%7D&peekPath=%2Fwandb-smle%2Fweave-cookboook-demo%2Fcalls%2F44678b3e-ebcf-481d-80de-7c4a578fd41d%3Ftracetree%3D1

# Define a list of patterns to search for
SEARCH_PATTERNS = ["model", "inputs", "output"]

def find_model_name_recursive(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            for pattern in SEARCH_PATTERNS:
                if pattern in key.lower():
                    for model_name in MODEL_NAMES:
                        if model_name == str(value):
                            return model_name
            result = find_model_name_recursive(value)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_model_name_recursive(item)
            if result:
                return result
    elif isinstance(obj, weave.trace.vals.WeaveObject):
        for attr in dir(obj):
            if not attr.startswith('_'):  # Skip private attributes
                value = getattr(obj, attr)
                for pattern in SEARCH_PATTERNS:
                    if pattern in attr.lower():
                        for model_name in MODEL_NAMES:
                            if model_name == str(value):
                                return model_name
                result = find_model_name_recursive(value)
                if result:
                    return result
    return None

# Function to process calls data
def process_calls(calls, trace_roots_only):
    records = []
    for call in calls:
        call_id = call.id
        model = "N/A"
        
        # Scan inputs and outputs for model name
        inputs_str = json.dumps(call.inputs)
        outputs_str = json.dumps(call.output)
        for model_name in MODEL_NAMES:
            if model_name in inputs_str or model_name in outputs_str:
                model = model_name
                break
        
        # If model is still N/A and trace_roots_only is True, recursively check for model name
        if model == "N/A" and trace_roots_only:
            model = find_model_name_recursive(call) or "N/A"
        
        # Extract costs
        costs = call.summary.get("weave", {}).get("costs", {})
        total_cost = 0.0
        total_tokens = 0
        for model_cost in costs.values():
            prompt_tokens = model_cost.get("prompt_tokens", 0)
            completion_tokens = model_cost.get("completion_tokens", 0)
            tokens = prompt_tokens + completion_tokens
            total_tokens += tokens

            prompt_cost = model_cost.get("prompt_tokens_total_cost", 0.0)
            completion_cost = model_cost.get("completion_tokens_total_cost", 0.0)
            total_cost += prompt_cost + completion_cost
        
        # Extract feedback
        feedback = call.summary.get("weave", {}).get("feedback", [])
        
        if isinstance(feedback, list):
            # Aggregate thumbs up and thumbs down from the list
            thumbs_up = sum(1 for item in feedback if isinstance(item, dict) and item.get("payload", {}).get("emoji") == "👍")
            thumbs_down = sum(1 for item in feedback if isinstance(item, dict) and item.get("payload", {}).get("emoji") == "👎")
        elif isinstance(feedback, dict):
            thumbs_up = feedback.get("thumbs_up", 0)
            thumbs_down = feedback.get("thumbs_down", 0)
        else:
            # Handle unexpected types
            st.warning(f"Unexpected feedback type for call {call_id}: {type(feedback)}")
            thumbs_up = 0
            thumbs_down = 0
        
        # Extract latency
        latency_ms = call.summary.get("weave", {}).get("latency_ms", 0)
        
        # Extract trace information
        trace_id = call.trace_id
        display_name = call.display_name
        inputs = call.inputs
        outputs = call.output
        
        # Extract started_at with safety
        started_at = getattr(call, 'started_at', None)
        if not started_at:
            started_at = datetime.min  # Assign a default value if missing
        
        records.append({
            "Call ID": call_id,
            "Trace ID": trace_id,
            "Display Name": display_name,
            "Model": model,
            "Tokens": total_tokens,
            "Cost (USD)": total_cost,
            "Latency (ms)": latency_ms,
            "Thumbs Up": thumbs_up,
            "Thumbs Down": thumbs_down,
            "Started At": started_at,
            "Inputs": json.dumps(inputs, default=str),
            "Outputs": json.dumps(outputs, default=str)
        })
    if not records:
        st.warning("No records processed.")
    df = pd.DataFrame(records)
    return df

# Function to calculate feedback summary
def calculate_feedback_summary(df):
    thumbs_up_total = df["Thumbs Up"].sum()
    thumbs_down_total = df["Thumbs Down"].sum()
    return thumbs_up_total, thumbs_down_total

# Function to plot feedback pie chart using Plotly
def plot_feedback_pie_chart(thumbs_up, thumbs_down):
    labels = ['Thumbs Up', 'Thumbs Down']
    values = [thumbs_up, thumbs_down]
    colors = ['#66b3ff', '#ff9999']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), hole=.3)])
    fig.update_traces(textinfo='percent+label', hoverinfo='label+percent')
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

# Function to plot model usage over time
def plot_model_usage(df):
    fig = px.area(df, x="Time", y="Total Usage (USD)", color="Model", title="Model Usage Over Time")
    fig.update_layout(xaxis_title="Time", yaxis_title="Total Cost (USD)", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# Function to plot latency over time
def plot_latency_over_time(df):
    fig = px.area(df, x="Time", y="Latency (ms)", color="Model", title="Latency Over Time")
    fig.update_layout(xaxis_title="Time", yaxis_title="Latency (ms)", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# Function to plot model cost distribution
def plot_model_cost_distribution(df):
    fig = px.bar(df, x="Model", y="Cost (USD)", color="Model", title="Cost Distribution by Model")
    fig.update_layout(xaxis_title="Model", yaxis_title="Cost (USD)")
    st.plotly_chart(fig, use_container_width=True)

# Main function to render the dashboard
def render_dashboard():
    st.markdown("<div class='header'>Weave LLM Monitoring Dashboard</div>", unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Add trace_roots_only toggle
    trace_roots_only = st.sidebar.toggle("Trace Roots Only", value=True)
    
    selected_project = st.sidebar.selectbox(
        "Select Weave Project",
        AVAILABLE_PROJECTS,
        index=0  # Default to the first project
    )

    # Initialize Weave client
    client = init_weave_client(selected_project)

    if client is None:
        st.stop()  # Stop execution if client failed to initialize

    # Define the time range (yesterday)
    end_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=1)

    st.write(f"Fetching data from **{start_time}** to **{end_time}** UTC for project **{selected_project}**.")
    st.write(f"Trace Roots Only: **{'Yes' if trace_roots_only else 'No'}**")

    # Fetch calls data
    with st.spinner("Fetching data from Weave..."):
        calls = fetch_calls(client, selected_project, start_time, end_time, trace_roots_only, limit=1000)
        if not calls:
            st.warning("No calls found for the selected time range.")
            return
        df_calls = process_calls(calls, trace_roots_only)
        st.success(f"Successfully fetched and processed {len(df_calls)} calls.")

    # Ensure 'Started At' column exists
    if 'Started At' not in df_calls.columns:
        st.error("The 'Started At' column is missing from the data.")
        return

    # Convert 'Started At' to datetime
    try:
        df_calls['Started At'] = pd.to_datetime(df_calls['Started At'])
    except Exception as e:
        st.error(f"Error converting 'Started At' to datetime: {e}")
        return

    # Display Metrics
    total_calls = len(df_calls)
    total_cost = df_calls["Cost (USD)"].sum()
    total_tokens = df_calls["Tokens"].sum()
    thumbs_up, thumbs_down = calculate_feedback_summary(df_calls)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls", total_calls)
    col2.metric("Total Cost (USD)", f"${total_cost:.6f}")  # Increased precision for cost
    col3.metric("Total Tokens", f"{total_tokens}")
    col4.metric("Feedback 👍 / 👎", f"{thumbs_up} / {thumbs_down}")

    st.markdown("---")

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Feedback Summary", "Cost Distribution", "Model Usage", "Latency"])

    with tab1:
        st.subheader("Feedback Summary")
        plot_feedback_pie_chart(thumbs_up, thumbs_down)
        st.markdown("### Detailed Feedback")
        st.dataframe(
            df_calls[["Call ID", "Model", "Thumbs Up", "Thumbs Down", "Started At"]]
            .sort_values(by="Started At", ascending=False)
            .reset_index(drop=True)
        )

    with tab2:
        st.subheader("Cost Distribution by Model")
        plot_model_cost_distribution(df_calls)
        st.markdown("### Detailed Cost Breakdown")
        df_cost_breakdown = df_calls.groupby("Model").agg({
            "Cost (USD)": "sum",
            "Tokens": "sum"
        }).reset_index().sort_values(by="Cost (USD)", ascending=False)
        st.dataframe(df_cost_breakdown)

    with tab3:
        st.subheader("Model Usage Over Time")
        # Aggregate cost over time by model
        df_usage = df_calls.groupby(['Started At', 'Model']).agg({'Cost (USD)': 'sum'}).reset_index()
        df_usage['Time'] = df_usage['Started At']
        df_usage['Total Usage (USD)'] = df_usage['Cost (USD)']
        plot_model_usage(df_usage)

    with tab4:
        st.subheader("Latency Over Time")
        # Aggregate latency over time by model
        df_latency = df_calls.groupby(['Started At', 'Model']).agg({'Latency (ms)': 'mean'}).reset_index()
        df_latency['Time'] = df_latency['Started At']
        if not df_latency.empty:
            plot_latency_over_time(df_latency)
        else:
            st.warning("No latency data available for the selected time range.")

    st.markdown("---")

    # Display raw data
    with st.expander("Show Raw Data"):
        # Display a more complete table including trace name, inputs, outputs
        raw_data = df_calls.copy()
        # Format Inputs and Outputs for better readability
        raw_data['Inputs'] = raw_data['Inputs'].apply(lambda x: x if isinstance(x, str) else json.dumps(x, indent=2))
        raw_data['Outputs'] = raw_data['Outputs'].apply(lambda x: x if isinstance(x, str) else json.dumps(x, indent=2))
        st.dataframe(raw_data[['Call ID', 'Trace ID', 'Display Name', 'Model', 'Tokens', 'Cost (USD)', 'Latency (ms)', 'Thumbs Up', 'Thumbs Down', 'Started At', 'Inputs', 'Outputs']].reset_index(drop=True))
        
        # Allow downloading the raw data as CSV
        csv = raw_data.to_csv(index=False)
        st.download_button(
            label="Download raw data as CSV",
            data=csv,
            file_name='raw_calls_data.csv',
            mime='text/csv',
        )

# Run the dashboard app
if __name__ == "__main__":
    render_dashboard()
