import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate

llm = Groq(model="llama3-70b-8192", api_key="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")

specific_products_df = pd.read_csv("data/demand_forecasting.csv")
inventory_audits = pd.read_csv("data/inventory_monitoring.csv")
pricing_and_market_data = pd.read_csv("data/pricing_optimization.csv")


instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

def store_level_trends(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.drop(columns=['Date'], inplace=True)
    
    agg_df = df.groupby("Store ID").agg(
        Total_Products=('Product ID', 'count'),
        Total_Sales=('Sales Quantity', 'sum'),
        Avg_Price=('Price', 'mean'),
        Prod_Sold_on_Promotion=('Promotions', lambda x: (x == 'Yes').sum()),
        Products_Festival=('Seasonality Factors', lambda x: (x == 'Festival').sum()),
        Products_Holiday=('Seasonality Factors', lambda x: (x == 'Holiday').sum()),
        Competitor_Affected_Prods=('External Factors', lambda x: (x == 'Competitor Pricing').sum()),
        Weather_Affected_Prods=('External Factors', lambda x: (x == 'Weather').sum()),
        High_Demand_Prods=('Demand Trend', lambda x: (x == 'Increasing').sum()),
        Regular_Customer=('Customer Segments', lambda x: (x == 'Regular').sum()),
        Premium_Customer=('Customer Segments', lambda x: (x == 'Premium').sum()),
        Budget_Customer=('Customer Segments', lambda x: (x == 'Budget').sum()),
        Max_Sale_Day=('Day', lambda x: x.mode()[0] if not x.mode().empty else None),
        Max_Sale_Month=('Month', lambda x: x.mode()[0] if not x.mode().empty else None)).reset_index()
    
    agg_df['Avg_Sales'] = (agg_df['Total_Sales'] / agg_df['Total_Products']).round().astype(int)
    agg_df['Avg_Price'] = agg_df['Avg_Price'].round(2)
    agg_df['Store ID'] = df['Store ID'].apply(lambda x: f"STORE-{x}")
    
    return agg_df

    
specific_products_query_engine = PandasQueryEngine(df=specific_products_df, verbose=True, instruction_str=instruction_str, llm=llm )
specific_products_query_engine.update_prompts({"pandas_prompt": new_prompt})

store_level_trends_query_engine = PandasQueryEngine(df=store_level_trends(specific_products_df), verbose=True, instruction_str=instruction_str, llm=llm )
store_level_trends_query_engine.update_prompts({"pandas_prompt": new_prompt})

inventory_audits_query_engine = PandasQueryEngine(df=inventory_audits, verbose=True, instruction_str=instruction_str, llm=llm )
inventory_audits_query_engine.update_prompts({"pandas_prompt": new_prompt})

pricing_and_market_performance_query_engine = PandasQueryEngine(df=pricing_and_market_data, verbose=True, instruction_str=instruction_str, llm=llm )
pricing_and_market_performance_query_engine.update_prompts({"pandas_prompt": new_prompt})

from llama_index.core.tools import FunctionTool
import os

note_file = os.path.join("data", "notes.txt")

def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([note + "\n"])

    return "note saved"


note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user",
)


# Update your tools list
tools = [
    QueryEngineTool(
        query_engine=store_level_trends_query_engine,
        metadata=ToolMetadata(
            name="aggregated_store_level_data",
            description="""Total Products, Average Sales, Average Price of Products, 
            Number of Products Sold With Promotions, Number of Products Sold in Festival, Number of Products Sold in Holiday,
            Number of Products affected by Competitors, Number of Products affected by Weather, Number of Products with Higher Demand, 
            Number of Regular Costumers, Number of Premium Costumers, Number of Budget Costumers, Dat of Maximum Sale, Month of Maximum Sale. """,
        ),
    ),
    QueryEngineTool(
        query_engine=specific_products_query_engine,
        metadata=ToolMetadata(
            name="specific_products_data",
            description="""This dataset contains sales transaction data with Product ID,
            Date, Store ID, Sales Quantity, Price, Promotions, Seasonality Factors, External Factors, Demand Trend,
            and Customer Segments.""",
        ),
    ),
    QueryEngineTool(
        query_engine=inventory_audits_query_engine,
        metadata=ToolMetadata(
            name="inventary_and_supply_chain_data",
            description="""This dataset provides inventory and supply chain information for individual products across stores. 
                           It includes Product ID, Store ID, Stock Levels, Supplier Lead Time, Stockout Frequency, Reorder Point,
                           Expiry Date, Warehouse Capacity, and Order Fulfillment Time. This data is useful for managing inventory, 
                           predicting stockouts, optimizing reorder points, and assessing supply chain efficiency..""",
        ),
    ),
    QueryEngineTool(
        query_engine=pricing_and_market_performance_query_engine,
        metadata=ToolMetadata(
            name="pricing_and_market_performance_query_data",
            description="""This dataset provides insights into product pricing, market competitiveness, and customer response. 
            It includes Product ID, Store ID, Price, Competitor Prices, Discounts, Sales Volume, Customer Reviews, Return Rate,
            Storage Cost, and Elasticity Index. This data is useful for analyzing price competitiveness, discount effectiveness, 
            customer satisfaction, and demand elasticity..""",
        ),
    ),
]


context = """Tool - inventary_and_supply_chain_data - 
             Purpose: 
             Use Inventory & Supply Chain Data for stock management, optimizing reorder strategies, reducing stockouts, and improving supply chain logistics.

             Tool - pricing_and_market_performance_query_data.
             Purpose:
             Use this dataset when analyzing pricing strategies, discount impacts, customer behavior, and market competitiveness.

             Tool - aggregated_store_level_data - 
             Purpose: 
             This data is useful for analyzing particular store performance, customer demographics, 
             and the impact of external factors on store-level sales. This dataset aggregates data at the 
             Store level, providing insights into Indivdual Store's Total Products, Total Sales, Average Price, 
             Product Performance (on Promotion, during Festivals, Holidays), External Influences 
             (Competitor and Weather Impact), and Customer Segments (Regular, Premium, Budget).
             
             Tool - specific_products_data
             Use this dataset when you need to focus on analyzing product sales trends, the effect of promotions 
             or external factors, and customer segments for specific products across stores. 
             This dataset contains detailed sales transactions for individual products across different stores. 
             It includes Product ID, Date of Sale, Store ID, Sales Quantity, Price, Promotions, Seasonality and 
             External Factors, Demand Trends, and Customer Segments. This data is ideal for analyzing product-level 
             sales trends, customer behavior, and the influence of external factors on product performance
             """

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)



from interactive import initialize_interactive_display
# Import necessary libraries (add these if not already imported)
import streamlit as st
import io
import re
from contextlib import redirect_stdout
from PIL import Image
import threading
import time
import queue
import json
import altair as alt
from datetime import datetime

import re
from dataclasses import dataclass
from typing import List, Dict, Optional

def format_agent_output(output):
    # Remove ANSI escape codes
    output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)
    
    # Define box formatting function
    def boxify(text):
        lines = text.split('\n')
        max_len = max(len(line) for line in lines)
        border = f"+{'-' * (max_len + 2)}+"
        boxed_text = '\n'.join([border] + [f"| {line.ljust(max_len)} |" for line in lines] + [border])
        return boxed_text
    
    # Patterns to identify Thought, Action, Action Input, Pandas Instructions, Pandas Output, and Observations
    patterns = {
        "Thought": r"(?<=Thought: )(.*?)(?=\n|$)",
        "Action": r"(?<=Action: )(.*?)(?=\n|$)",
        "Action Input": r"(?<=Action Input: )(.*?)(?=\n|$)",
        "Pandas Instructions": r"(?=> Pandas Instructions:\n)(.*?)(?=> Pandas Output:|Observation:|Thought:|$)",
        "Pandas Output": r"(?=> Pandas Output:\n)(.*?)(?=Observation:|Thought:|$)",
        "Observation": r"(?<=Observation: )(.*?)(?=\n|$)",
    }
    
    formatted_output = ""
    for label, pattern in patterns.items():
        matches = re.findall(pattern, output, re.DOTALL)
        for match in matches:
            text = f"{label}:\n{match.strip()}"  # Construct the string first
            formatted_output += f"\n{boxify(text)}\n\n"
    
    return formatted_output.strip()

def parse_agent_output(output):
    # Remove ANSI escape codes
    output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)

    # Remove "Running step" lines
    output = re.sub(r'> Running step [a-f0-9-]+\. Step input: .*', '', output)

    return output.strip()
    
# Set page configuration
st.set_page_config(layout="wide", page_title="Nishant-Accenture Analytics Dashboard")

# Initialize session state for agent thinking
if "agent_thinking" not in st.session_state:
    st.session_state["agent_thinking"] = []

    
st.markdown("""
    <style>
        .block-container {
            padding-bottom: 0px !important;
        }
        hr {
            margin-top: 5px !important;
            margin-bottom: 5px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load and display an image
image = Image.open("image.png")
st.image(image, use_container_width=True)

# Define the layout with two fixed-size containers
left_col, right_col = st.columns([1, 1])


with left_col:
    st.header("Accenture AI")
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Create a container with fixed height for chat history
    chat_container = st.container(height=350, border=True)
    
    # Display all messages inside the fixed-height container
    with chat_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # User input outside the scrolling chat container
    if prompt := st.chat_input("What is up?"):
        # Add user message to session state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        # Clear previous agent thinking
        st.session_state["agent_thinking"] = []
        
        # Process response
        with st.spinner("Thinking..."):
            # Create a buffer to capture stdout
            buffer = io.StringIO()
            
            # Redirect stdout to our buffer and run the agent
            with redirect_stdout(buffer):
                response = agent.query(prompt)
            
            # Get the captured output and log it
            captured_output = buffer.getvalue()
            structured_steps = parse_agent_output(captured_output)
            st.session_state["agent_thinking"] = structured_steps.splitlines()
        
        # Add assistant response to session state
        st.session_state["messages"].append({"role": "assistant", "content": response})
        
        # Force rerun to update the chat history
        st.rerun()


# Right column - Agent Thinking Process 
with right_col:
    initialize_interactive_display()
    # st.header("Reasoning")
    
    # # Create a container for the agent thinking with same height as the chat
    # thinking_container = st.container(height=350, border=True)
    
    # with thinking_container:
    #     if "agent_thinking" in st.session_state and st.session_state["agent_thinking"]:
    #         # Create tabs for different views of the thinking process
    #         tab1, tab2, tab3 = st.tabs(["Raw Output", "Thoughts", "Actions"])
            
    #         with tab1:
    #             # Display the raw thinking process
    #             thinking_text = "\n".join(st.session_state["agent_thinking"])
    #             with st.text_area("Complete Process", thinking_text, height=300):
    #                 # Display interactive visualization
    #                 display_interactive_agent_output(thinking_text)
            
    #         with tab2:
    #             # Display only the thoughts
    #             thoughts = [line for line in st.session_state["agent_thinking"] if "Thought:" in line]
    #             if thoughts:
    #                 for thought in thoughts:
    #                     st.markdown(f"**{thought}**")
    #             else:
    #                 st.info("No thoughts captured in the current process.")
            
    #         with tab3:
    #             # Display actions and their inputs
    #             actions = [line for line in st.session_state["agent_thinking"] if line.startswith("Action:")]
    #             action_inputs = [line for line in st.session_state["agent_thinking"] if line.startswith("Action Input:")]
                
    #             if actions:
    #                 for i, action in enumerate(actions):
    #                     st.markdown(f"**{action}**")
    #                     if i < len(action_inputs):
    #                         st.code(action_inputs[i])
    #             else:
    #                 st.info("No actions captured in the current process.")
    #     else:
    #         st.info("Agent reasoning will appear here when you ask a question.")
    

# Footer
st.markdown("---")
st.markdown("© 2025 Retail Analytics Dashboard | Powered by Accenture AI | Logout (nishant0363)")