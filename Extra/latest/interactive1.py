import streamlit as st
import re
import pandas as pd
import json
import altair as alt
from datetime import datetime
import markdown
import base64
from io import StringIO

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
                # Try to parse DataFrames from Pandas Output
                if component == "Observation" and "rows" in content and "columns" in content:
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

import pandas as pd
import streamlit as st
from io import StringIO

def display_interactive_agent_output(thinking_text):
    """Create an Interactive display for agent reasoning using only native Streamlit components."""
    
    # Parse structured steps
    steps = parse_cleaned_output(thinking_text)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Step-by-Step", "Overview", "History"])
    
    # Store the content for the history
    history_content = []
    
    with tab1:
        # Create an expandable view for each reasoning step
        for i, step in enumerate(steps):
            step_title = f"Step {step['id']}: "
            
            with st.expander(step_title, expanded=(i==i)):
                cols = st.columns([5, 1])
                
                # Left column for step content
                with cols[0]:
                    step_content = []  # Collect content for history
                    
                    if "Thought" in step:
                        st.markdown("#### Thought")
                        st.markdown(step["Thought"])
                        step_content.append(f"#### Thought\n{step['Thought']}")
                    
                    if "Action" in step:
                        st.markdown("#### Action")
                        st.markdown(f"**{step['Action']}**")
                        step_content.append(f"#### Action\n**{step['Action']}**")
                    
                    if "Action Input" in step:
                        st.markdown("#### Action Input")
                        st.code(step["Action Input"], language="json")
                        step_content.append(f"#### Action Input\n```json\n{step['Action Input']}\n```")
                    
                    if "Pandas Instructions" in step:
                        st.markdown("#### Pandas Instructions")
                        st.code(step["Pandas Instructions"], language="python")
                        step_content.append(f"#### Pandas Instructions\n```python\n{step['Pandas Instructions']}\n```")
                    
                    if "Pandas Output" in step and step.get("has_dataframe", False):
                        st.markdown("#### Pandas Output")
                        try:
                            df_text = step["Pandas Output"]
                            
                            # Remove extra metadata like "[5 rows x 16 columns]"
                            df_lines = df_text.strip().split("\n")
                            if "[" in df_lines[-1] and "]" in df_lines[-1]:  # Detect metadata row
                                df_lines = df_lines[:-1]  # Remove it
                            
                            clean_text = "\n".join(df_lines)
                            
                            # Convert to DataFrame
                            df = pd.read_csv(StringIO(clean_text), sep=r"\s{2,}", engine="python")  # Handle whitespace separation
                            st.dataframe(df)  # Display DataFrame neatly
                            step_content.append(f"#### Pandas Output\n```\n{clean_text}\n```")
                        except Exception as e:
                            st.text(step["Pandas Output"])  # Fallback if parsing fails
                            step_content.append(f"#### Pandas Output\n```\n{step['Pandas Output']}\n```")
                            
                    elif "Pandas Output" in step:
                        st.markdown("#### Pandas Output")
                        st.text(step["Pandas Output"])
                        step_content.append(f"#### Pandas Output\n```\n{step['Pandas Output']}\n```")
                        
                    if "Observation" in step and step.get("has_dataframe", False):
                        st.markdown("#### Observation")
                        try:
                            df_text = step["Observation"]
                            
                            # Remove extra metadata like "[5 rows x 16 columns]"
                            df_lines = df_text.strip().split("\n")
                            if "[" in df_lines[-1] and "]" in df_lines[-1]:  # Detect metadata row
                                df_lines = df_lines[:-1]  # Remove it
                            
                            clean_text = "\n".join(df_lines)
                            
                            # Convert to DataFrame
                            df = pd.read_csv(StringIO(clean_text), sep=r"\s{2,}", engine="python")  # Handle whitespace separation
                            st.dataframe(df)  # Display DataFrame neatly
                            step_content.append(f"#### Observation\n```\n{clean_text}\n```")
                        except Exception as e:
                            st.text(step["Observation"])  # Fallback if parsing fails
                            step_content.append(f"#### Observation\n```\n{step['Observation']}\n```")
                    
                    elif "Observation" in step:
                        st.markdown("#### Observation")
                        if "Error" in step["Observation"]:
                            st.error(step["Observation"])
                            step_content.append(f"#### Observation (Error)\n{step['Observation']}")
                        else:
                            st.info(step["Observation"])
                            step_content.append(f"#### Observation\n{step['Observation']}")
                    
                    if "Answer" in step:
                        st.markdown("#### Final Answer")
                        st.success(step["Answer"])
                        step_content.append(f"#### Final Answer\n{step['Answer']}")
                
                # Add step content to history
                if step_content:
                    history_content.append({
                        "title": step_title,
                        "content": "\n\n".join(step_content)
                    })
    
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
        # History tab that accumulates and saves content
        st.markdown("### Agent Reasoning History")
        
        # Import datetime at the beginning of the function
        from datetime import datetime
        
        # Session state to store history across reruns
        if 'agent_history' not in st.session_state:
            st.session_state.agent_history = []
        
        # Update session state with new history if it has content and isn't already there
        if history_content:
            # Create a hash of the current content to check if it's new
            import hashlib
            content_hash = hashlib.md5(str(history_content).encode()).hexdigest()
            
            if 'last_content_hash' not in st.session_state or st.session_state.last_content_hash != content_hash:
                # Instead of replacing, we'll append the new content
                run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Add a timestamp separator for this run
                if st.session_state.agent_history:
                    st.session_state.agent_history.append({
                        "title": f"--- New Run ({run_timestamp}) ---",
                        "content": f"New agent execution at {run_timestamp}"
                    })
                
                # Append the new content to existing history
                st.session_state.agent_history.extend(history_content)
                st.session_state.last_content_hash = content_hash     
                        
        # If we have a history, display it with expandable sections
        if st.session_state.agent_history:
            # Create full markdown content for export
            full_history_markdown = ""
            
            for i, item in enumerate(st.session_state.agent_history):
                with st.expander(item["title"], expanded=False):
                    st.markdown(item["content"])
                
                # Add to the full markdown content
                full_history_markdown += f"## {item['title']}\n\n{item['content']}\n\n---\n\n"
            
            # Add download buttons
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # HTML download option
            st.download_button(
                label="Download as HTML",
                data=f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Agent Reasoning History</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }}
                        h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                        h2 {{ color: #3498db; margin-top: 30px; }}
                        h4 {{ color: #7f8c8d; margin-top: 20px; margin-bottom: 5px; }}
                        pre {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                        hr {{ border: 0; height: 1px; background: #eee; margin: 30px 0; }}
                        .success {{ background-color: #e8f8f5; padding: 10px; border-left: 5px solid #2ecc71; }}
                        .info {{ background-color: #eaf2f8; padding: 10px; border-left: 5px solid #3498db; }}
                        .error {{ background-color: #fadbd8; padding: 10px; border-left: 5px solid #e74c3c; }}
                    </style>
                </head>
                <body>
                    <h1>Agent Reasoning History</h1>
                    {full_history_markdown.replace("```", "<pre>").replace("```", "</pre>")}
                </body>
                </html>
                """,
                file_name=f"agent_reasoning_history_{timestamp}.html",
                mime="text/html"
            )
            
            # Markdown download option
            st.download_button(
                label="Download as Markdown",
                data=full_history_markdown,
                file_name=f"agent_reasoning_history_{timestamp}.md",
                mime="text/markdown"
            )
            
            # Add clear history button
            if st.button("Clear History"):
                st.session_state.agent_history = []
                st.session_state.last_content_hash = ""
                st.rerun()
        else:
            st.info("No history recorded yet. Run the agent to see results here.")

# Function for use in main file
def initialize_interactive_display():
        
    # Main column - Agent Thinking Process
    with st.container():
        st.header("Agent Reasoning")
        
        # Create a container for the agent thinking
        thinking_container = st.container(height=480, border=True)
        
        with thinking_container:
            if "agent_thinking" in st.session_state and st.session_state["agent_thinking"]:
                # Get the raw thinking text
                thinking_text = "\n".join(st.session_state["agent_thinking"])

                # with st.expander("Raw Output", expanded=False):
                #     st.text_area("Complete Process", thinking_text, height=300)
                
                # Display interactive visualization
                display_interactive_agent_output(thinking_text)
                with st.expander("Raw Output", expanded=False):
                    st.text_area("Complete Process", thinking_text, height=300)
            else:
                st.info("Interactive display for agent reasoning. Run a query to see the agent's thinking process.")