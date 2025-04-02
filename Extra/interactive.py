import streamlit as st
import re
import pandas as pd
import json
import altair as alt
from datetime import datetime

def parse_cleaned_output(output):
    """Parse the agent output to extract structured information."""
    # Remove ANSI escape codes
    output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)
    
    # Define regex patterns to extract different components
    patterns = {
        "Thought": r"Thought: (.*?)(?=\nAction:|$)",
        "Action": r"Action: (.*?)(?=\nAction Input:|$)",
        "Action Input": r"Action Input: (.*?)(?=\n>|$)",
        "Pandas Instructions": r"> Pandas Instructions:\n(.*?)(?=\n> Pandas Output:|$)",
        "Pandas Output": r"> Pandas Output:(.*?)(?=\nObservation:|$)",
        "Observation": r"Observation: (.*?)(?=\nThought:|$)",
        "Answer": r"Answer: (.*?)(?=\n|$)"
    }
    
    # Extract all instances of each pattern
    steps = []
    step_counter = 0
    
    # Find all occurrences of thought patterns to use as separators
    thoughts = re.finditer(r"Thought: (.*?)(?=\nAction:|$)", output, re.DOTALL)
    thought_positions = [match.start() for match in thoughts]
    
    # Add the end position
    thought_positions.append(len(output))
    
    # Process each thought block
    for i in range(len(thought_positions) - 1):
        step_counter += 1
        step = {"id": step_counter, "timestamp": datetime.now().strftime("%H:%M:%S")}
        
        block = output[thought_positions[i]:thought_positions[i+1]]
        
        # Extract components from this block
        for component, pattern in patterns.items():
            matches = re.search(pattern, block, re.DOTALL)
            if matches:
                content = matches.group(1).strip()
                step[component] = content
                
                # Try to parse JSON from Action Input
                if component == "Action Input" and content.startswith('{'):
                    try:
                        step["Action Input Data"] = json.loads(content)
                    except:
                        pass
                
                # Try to parse DataFrames from Pandas Output
                if component == "Pandas Output" and "rows" in content and "columns" in content:
                    step["has_dataframe"] = True
        
        steps.append(step)
    
    # Find final answer if it exists
    answer_match = re.search(r"Answer: (.*?)(?=\n|$)", output, re.DOTALL)
    if answer_match:
        steps.append({
            "id": step_counter + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "Answer": answer_match.group(1).strip()
        })
    
    return steps

def display_interactive_agent_output(thinking_text):
    """Create an interactive display for agent reasoning using only native Streamlit components."""
    
    # Parse structured steps
    steps = parse_cleaned_output(thinking_text)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Step-by-Step", "Overview", "Visualizations"])
    
    with tab1:
        # Create an expandable view for each reasoning step
        for i, step in enumerate(steps):
            step_title = f"Step {step['id']}: "
            if "Thought" in step:
                # Truncate thought for display in expander title
                thought_preview = step["Thought"][:60] + "..." if len(step["Thought"]) > 60 else step["Thought"]
                step_title += thought_preview
            elif "Answer" in step:
                step_title = "✅ Final Answer"
            
            with st.expander(step_title, expanded=(i==0)):
                cols = st.columns([3, 1])
                
                # Left column for step content
                with cols[0]:
                    if "Thought" in step:
                        st.markdown("### 💭 Thought")
                        st.markdown(step["Thought"])
                    
                    if "Action" in step:
                        st.markdown("### 🔄 Action")
                        st.markdown(f"**{step['Action']}**")
                    
                    if "Action Input" in step:
                        st.markdown("### ⌨️ Action Input")
                        st.code(step["Action Input"], language="json")
                    
                    if "Pandas Instructions" in step:
                        st.markdown("### 📊 Pandas Instructions")
                        st.code(step["Pandas Instructions"], language="python")
                    
                    if "Pandas Output" in step and step.get("has_dataframe", False):
                        st.markdown("### 📈 Pandas Output")
                        # Try to convert text output to a DataFrame
                        try:
                            # Simple approach to parse DataFrame text
                            df_text = step["Pandas Output"]
                            # Format the DataFrame text more nicely
                            st.text(df_text)
                            # If you had a function to parse this into an actual DataFrame, you could display it
                            # st.dataframe(df)
                        except:
                            st.text(step["Pandas Output"])
                    elif "Pandas Output" in step:
                        st.markdown("### 📈 Pandas Output")
                        st.text(step["Pandas Output"])
                    
                    if "Observation" in step:
                        st.markdown("### 👁️ Observation")
                        if "Error" in step["Observation"]:
                            st.error(step["Observation"])
                        else:
                            st.info(step["Observation"])
                    
                    if "Answer" in step:
                        st.markdown("### ✅ Final Answer")
                        st.success(step["Answer"])
                
                # Right column for metadata and mini-visualization
                with cols[1]:
                    st.metric("Step", step["id"])
                    if "Action" in step:
                        # Display action type
                        st.markdown("##### Action Type")
                        st.code(step["Action"])
                    
                    # Add a timestamp
                    st.markdown("##### Time")
                    st.text(step.get("timestamp", ""))
    
    with tab2:
        # Overview tab with metrics and summary
        st.markdown("### Agent Reasoning Overview")
        
        # Add metrics row
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Total Steps", len(steps))
        with metrics_cols[1]:
            action_count = sum(1 for step in steps if "Action" in step)
            st.metric("Actions Taken", action_count)
        with metrics_cols[2]:
            error_count = sum(1 for step in steps if "Observation" in step and "Error" in step["Observation"])
            st.metric("Errors", error_count)
        with metrics_cols[3]:
            has_answer = any("Answer" in step for step in steps)
            st.metric("Completed", "Yes" if has_answer else "No")
        
        # Create a concise summary table
        step_summary = []
        for step in steps:
            summary_row = {
                "Step": step["id"],
                "Type": "Answer" if "Answer" in step else "Action" if "Action" in step else "Thought",
                "Content": step.get("Answer", step.get("Action", step.get("Thought", "")[:50] + "...")),
                "Status": "Error" if "Observation" in step and "Error" in step["Observation"] else "Success"
            }
            step_summary.append(summary_row)
        
        st.markdown("### Step Summary")
        summary_df = pd.DataFrame(step_summary)
        st.dataframe(
            summary_df,
            column_config={
                "Step": st.column_config.NumberColumn("Step", format="%d"),
                "Type": st.column_config.TextColumn("Type"),
                "Content": st.column_config.TextColumn("Content"),
                "Status": st.column_config.TextColumn("Status")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Show final answer if available
        final_answer = next((step["Answer"] for step in steps if "Answer" in step), None)
        if final_answer:
            st.markdown("### Final Answer")
            st.success(final_answer)
    
    with tab3:
        # Visualizations tab
        st.markdown("### Agent Reasoning Visualizations")
        
        # Create action flow visualization with Altair
        if steps:
            flow_data = []
            for i, step in enumerate(steps):
                if "Action" in step:
                    flow_data.append({
                        "Step": i+1,
                        "Action": step["Action"],
                        "Result": "Error" if ("Observation" in step and "Error" in step["Observation"]) else "Success"
                    })
            
            if flow_data:
                st.markdown("#### Action Flow")
                flow_df = pd.DataFrame(flow_data)
                
                # Create Altair chart
                action_chart = alt.Chart(flow_df).mark_circle(size=100).encode(
                    x=alt.X('Step:O', title='Step Number'),
                    y=alt.Y('Action:N', title='Action Type'),
                    color=alt.Color('Result:N', scale=alt.Scale(
                        domain=['Success', 'Error'],
                        range=['#28a745', '#dc3545']
                    )),
                    tooltip=['Step', 'Action', 'Result']
                ).properties(
                    width=600,
                    height=300,
                    title='Agent Action Flow'
                )
                
                st.altair_chart(action_chart, use_container_width=True)
                
                # Add a second visualization - action counts by type
                action_counts = flow_df['Action'].value_counts().reset_index()
                action_counts.columns = ['Action', 'Count']
                
                action_bar = alt.Chart(action_counts).mark_bar().encode(
                    x=alt.X('Action:N', title='Action Type'),
                    y=alt.Y('Count:Q', title='Number of Calls'),
                    color=alt.Color('Action:N', legend=None),
                    tooltip=['Action', 'Count']
                ).properties(
                    width=600,
                    height=300,
                    title='Action Type Distribution'
                )
                
                st.altair_chart(action_bar, use_container_width=True)
            else:
                st.info("No action data available for visualization")
        
        # Add a simple progress visualization
        st.markdown("#### Reasoning Progress")
        
        # Create a sequential progress visualization
        progress_cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(progress_cols, steps)):
            with col:
                if "Answer" in step:
                    st.markdown("✅")
                elif "Observation" in step and "Error" in step["Observation"]:
                    st.markdown("❌")
                elif "Action" in step:
                    st.markdown("🔄")
                else:
                    st.markdown("💭")
                st.write(f"Step {step['id']}")

# Function for use in main file
def initialize_interactive_display():
        
    # Main column - Agent Thinking Process
    with st.container():
        st.header("Agent Reasoning")
        
        # Create a container for the agent thinking
        thinking_container = st.container(height=350, border=True)
        
        with thinking_container:
            if "agent_thinking" in st.session_state and st.session_state["agent_thinking"]:
                # Get the raw thinking text
                thinking_text = "\n".join(st.session_state["agent_thinking"])

                with st.expander("Raw Output", expanded=False):
                    st.text_area("Complete Process", thinking_text, height=300)
                
                # Display interactive visualization
                display_interactive_agent_output(thinking_text)
            else:
                st.info("No agent reasoning data available yet. Run a query to see the agent's thinking process.")