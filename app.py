import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
from utils.predict_task_duration import TaskDurationPredictor
from utils.config import MODEL_PATH, ENVIRONMENT_MAPPING
from utils.llm import ReportGenerator
from utils.config import OPENAI_API_KEY

# Configure page
st.set_page_config(
    page_title="Task Manager",
    page_icon="🗓",
    # layout="wide"
)

# Initialize session state for tasks
if "tasks" not in st.session_state:
    st.session_state.tasks = pd.DataFrame(columns=[
        "Task ID", "Team Size", "Resource Availability", "Complexity", "Priority", "Risk",
        "Environment", "Dependencies", "Expected Total Task Duration", "Predicted Total Task Duration",
        "Expected Start Date", "Expected End Date", "Predicted Start Date", "Predicted End Date"
    ])

st.title("Task Manager")

# Create tabs
tab1, tab2 = st.tabs(["Upload CSV", "Manual Entry"])

with tab1:
    # Add file uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        # Read the CSV file
        try:
            df = pd.read_csv(uploaded_file)
            # Process the dataframe using TaskDurationPredictor
            processed_df = TaskDurationPredictor(MODEL_PATH, ENVIRONMENT_MAPPING).process_dataframe(df)
            st.session_state.tasks = processed_df
            st.success("CSV file uploaded and processed successfully!")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")





# Display the current tasks
if not st.session_state.tasks.empty:
    st.markdown("---")  # Add a separator line
    st.subheader("Tasks")
    st.dataframe(st.session_state.tasks)
    
    # Create two columns for buttons
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        show_gantt = st.button("Analysis and Insights", use_container_width=True)
    
    with button_col2:
        # Convert dataframe to CSV for download
        csv = st.session_state.tasks.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="updated_tasks.csv",
            mime="text/csv",
            use_container_width=True
        )

    if show_gantt:
        # Create Gantt chart using plotly.graph_objects
        df_gantt = []

        for idx, row in st.session_state.tasks.iterrows():
            # Add bar for Expected dates
            df_gantt.append(dict(
                Task=row['Task ID'],
                Start=row['Expected Start Date'],
                Finish=row['Expected End Date'],
                Resource='Expected Date'
            ))
            
            df_gantt.append(dict(
                Task=row['Task ID'],
                Start=row['Predicted Start Date'],
                Finish=row['Predicted End Date'],
                Resource='Predicted Date'
            ))
            
        # Create Gantt chart
        colors = {
        'Expected Date': 'rgb(46, 137, 205)',  # Blue
        'Predicted Date': 'rgb(198, 47, 105)'  # Pink
        }

        # Add bar for Predicted dates
        fig = ff.create_gantt(df_gantt, 
                        colors=colors,
                        index_col='Resource',
                        show_colorbar=True,
                        group_tasks=True,
                        showgrid_x=True,
                        showgrid_y=True)
        
        fig.update_traces(opacity=0.7, selector={'fill':'toself'})

        # Update layout
        fig.update_layout(
            title={
                'text': "Project Tasks Gantt Chart",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Date",
            yaxis_title="Task ID",
            height=400,
            barmode='overlay',
            showlegend=True,
            xaxis={
                'type': 'date',
                'tickformat': '%d/%m/%Y'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if OPENAI_API_KEY:
            st.markdown("---")  # Add a separator line
            st.markdown(ReportGenerator().generate_report(st.session_state.tasks))