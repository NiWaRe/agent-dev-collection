import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import weave

# Set the page layout to wide
st.set_page_config(layout="wide")

# Custom CSS to add padding to the sides of the app
st.markdown("""
    <style>
    .main {
        padding-left: 50px;
        padding-right: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

def init_weave():
    client = weave.init('wandb-smle/weave-rag-experiments')

@st.cache_resource
def get_weave_model(weave_url: str):
    model = weave.ref(weave_url).get()
    return model

# TODO: backend not functonal yet to retrieve costs
def get_costs_for_project(project_names: list[str]):
    total_cost = 0
    requests = 0
    call_total = 0
    for project_name in project_names:
        client = weave.init(project_name)
        calls = list(
            client.calls(filter={"trace_roots_only": True}, include_costs=True)
        )

        for call in calls:
            if hasattr(call, 'summary') and call.summary.get("weave") is not None:
                for k, cost in call.summary["weave"]["costs"].items():
                    requests += cost["requests"]
                    total_cost += cost["prompt_tokens_cost"]
                    total_cost += cost["completion_tokens_cost"]

        call_total += len(calls)

# Function to fetch data for the scores and model usage
def fetch_dashboard_data():
    # Simulated data for Scores
    scores_data = pd.DataFrame({
        "Name": [
            "gpt-4-mini", 
            "text-embedding-ada-002", 
        ],
        "Sum Tokens [#]": [1221, 239],
        "Sum Cost [USD]": [0.01, 0.70],
        "0": [2, 17],  # Thumbs-down reviews
        "1": [18, 43]   # Thumbs-up reviews
    })
    
    # Simulated data for Model Usage
    model_usage_data = pd.DataFrame({
        "Time": pd.date_range(start='2024-09-03 18:00', periods=24, freq='H'),
        "gpt-4-mini": np.random.rand(24) * 0.003,
        "text-embedding-ada-002": np.random.rand(24) * 0.001
    })
    
    return scores_data, model_usage_data

# Function to plot the feedback pie chart
def plot_feedback_pie_chart(scores_data):
    thumbs_up_total = scores_data["1"].sum()
    thumbs_down_total = scores_data["0"].sum()

    # Prepare data for the pie chart
    labels = ['Thumbs Up', 'Thumbs Down']
    sizes = [thumbs_up_total, thumbs_down_total]
    colors = ['#66b3ff', '#ff9999']

    # Plot the pie chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.05, 0.05))
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is drawn as a circle.
    
    st.pyplot(fig)

# Main function to render the dashboard
def render_dashboard():
    # Fetch the data
    scores_data, model_usage_data = fetch_dashboard_data()

    # Create two columns for the pie chart and the scores table
    col1, col2 = st.columns([1, 3])

    # Column 1: Pie Chart Section
    with col1:
        st.subheader("Feedback Summary")
        plot_feedback_pie_chart(scores_data)

    # Column 2: Scores Section
    with col2:
        st.subheader("Scores")
        st.table(scores_data)

    # Model Usage Graph with Summary Line
    st.subheader("Model Usage")

    # Calculate the total sum of the usage from all models, excluding the 'Time' column
    model_usage_data['Total'] = model_usage_data[['gpt-4-mini', 'text-embedding-ada-002']].sum(axis=1)

    # Plot the data including the total usage as a summary line
    st.line_chart(model_usage_data.set_index("Time"))

# Run the dashboard app
if __name__ == "__main__":
    st.title("Dashboard")
    render_dashboard()
