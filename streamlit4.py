import os
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
from llama_index.core import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer

chat_memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
llm1 = Groq(model="gemma2-9b-it", api_key="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")
llm2 = Groq(model="llama3-70b-8192", api_key="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")


specific_products_df = pd.read_csv("data/demand_forecasting.csv")
inventory_audits = pd.read_csv("data/inventory_monitoring.csv")
pricing_and_market_data = pd.read_csv("data/pricing_optimization.csv")



instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. Ensure the code correctly handles edge cases:
       - If multiple rows match the condition, return all relevant values as a list.
       - If the query involves finding a maximum/minimum, return all rows that share the same extreme value.
       - If NaN values are present, assume they should be ignored unless explicitly stated.
    3. The final line of code should be a Python expression that can be evaluated using `eval()`.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression.
    6. If the query is ambiguous, assume the most general and useful interpretation.
"""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`. df_description = 
    df has the following columns:
    
    Product ID: Unique identifier for each product. Example values are 4277, 5540, 5406.
    Date: The date of the sales transaction in YYYY-MM-DD format. Example values are 2024-01-03, 2024-04-29, 2024-01-11.
    Store ID: Unique identifier for the store where the sale occurred. Example values are 48, 10, 67.
    Sales Quantity: Number of units sold for the product on the given date. Example values are 330, 334, 429.
    Price: The selling price of the product per unit. Example values are 24.38, 74.98, 24.83.
    Promotions: Indicates whether the product was under a promotional offer ("Yes") or not ("No"). Example values are Yes, No.
    Seasonality Factors: Events or seasons affecting sales, such as holidays or festivals. Example values are Festival, Holiday, NaN.
    External Factors: Market conditions affecting sales, such as competitor pricing, economic indicators, or weather conditions. Example values are Competitor Pricing, Weather, Economic Indicator, NaN.
    Demand Trend: Describes the sales trend for the product over time, whether it is increasing, stable, or decreasing. Example values: Increasing, Stable, Decreasing.
    Customer Segments: The target customer group for the product, such as Regular, Premium, or Budget. Example values: Regular, Premium, Budget.

    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)
    
specific_products_query_engine = PandasQueryEngine(df=specific_products_df, verbose=True, instruction_str=instruction_str, llm=llm2 )
specific_products_query_engine.update_prompts({"pandas_prompt": new_prompt})



instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. Ensure the code correctly handles edge cases:
       - If multiple rows match the condition, return all relevant values as a list.
       - If the query involves finding a maximum/minimum, return all rows that share the same extreme value.
       - If NaN values are present, assume they should be ignored unless explicitly stated.
    3. The final line of code should be a Python expression that can be evaluated using eval().
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression.
    6. If the query is ambiguous, assume the most general and useful interpretation.
"""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python. The name of the dataframe is df. df_description = 
    df has the following columns:
    
    Product ID: Unique identifier for each product. Example values are 9286, 2605, 2859.
    Store ID: Unique identifier for the store where the product is stocked. Example values are 16, 60, 55.
    Stock Levels: The current inventory level of the product in the store. Example values are 700, 82, 145.
    Supplier Lead Time (days): The number of days required for suppliers to deliver restocked products. Example values are 10, 11, 25.
    Stockout Frequency: The frequency at which the product runs out of stock. Example values are 14, 1, 14.
    Reorder Point: The stock level at which a new order should be placed to prevent stockouts. Example values are 132, 127, 192.
    Expiry Date: The expiration date of the product in YYYY-MM-DD format. Example values are 2024-01-15, 2024-12-16, 2024-04-30.
    Warehouse Capacity: The storage capacity of the warehouse where the product is kept. Example values are 1052, 1262, 1457.
    Order Fulfillment Time (days): The number of days required to fulfill an order after placement. Example values are 6, 9, 12.

    This is the result of print(df.head()):
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)


inventory_audits_query_engine = PandasQueryEngine(df=inventory_audits, verbose=True, instruction_str=instruction_str, llm=llm2 )
inventory_audits_query_engine.update_prompts({"pandas_prompt": new_prompt})


instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. Ensure the code correctly handles edge cases:
       - If multiple rows match the condition, return all relevant values as a list.
       - If the query involves finding a maximum/minimum, return all rows that share the same extreme value.
       - If NaN values are present, assume they should be ignored unless explicitly stated.
    3. The final line of code should be a Python expression that can be evaluated using eval().
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression.
    6. If the query is ambiguous, assume the most general and useful interpretation.
"""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python. The name of the dataframe is df. df_description = 
    df has the following columns:
    
    Product ID: Unique identifier for each product. Example values are 9502, 2068, 7103.
    Store ID: Unique identifier for the store selling the product. Example values are 13, 77, 59.
    Price: The selling price of the product per unit. Example values are 31.61, 35.51, 6.54.
    Competitor Prices: The price of the same product sold by competitors. Example values are 56.14, 63.04, 30.61.
    Discounts: The percentage or absolute discount applied to the product. Example values are 19.68, 16.88, 10.86.
    Sales Volume: The number of units sold. Example values are 255, 5, 184.
    Customer Reviews: The average rating given by customers. Example values are 3, 3, 3.
    Return Rate (%): The percentage of units returned by customers. Example values are 13.33, 1.50, 9.44.
    Storage Cost: The cost of storing the product in a warehouse. Example values are 6.72, 8.38, 3.86.
    Elasticity Index: A measure of how demand responds to changes in price. Example values are 1.78, 1.67, 2.46.

    This is the result of print(df.head()):
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)



pricing_and_market_performance_query_engine = PandasQueryEngine(df=pricing_and_market_data, verbose=True, instruction_str=instruction_str, llm=llm2 )
pricing_and_market_performance_query_engine.update_prompts({"pandas_prompt": new_prompt})

from llama_index.core.tools import FunctionTool
import os


# Tools list
tools = [
    QueryEngineTool(
        query_engine=specific_products_query_engine,
        metadata=ToolMetadata(
            name="specific_products_data",
            description="""This dataset contains sales transaction data with Product ID,
            Date, Store ID, Sales Quantity, Price, Promotions, Seasonality Factors, External Factors, Demand Trend,
            and Customer Segments.
            It includes 
            [Product ID: Unique identifier for each product. Example values are 4277, 5540, 5406],
            [Date: The date of the sales transaction in YYYY-MM-DD format. Example values are 2024-01-03, 2024-04-29, 2024-01-11],
            [Store ID: Unique identifier for the store where the sale occurred. Example values are 48, 10, 67]
            [Sales Quantity: Number of units sold for the product on the given date. Example values are 330, 334, 429]
            [Price: The selling price of the product per unit. Example values are 24.38, 74.98, 24.83]
            [Promotions: Indicates whether the product was under a promotional offer ("Yes") or not ("No"). Example values are Yes, No]
            [Seasonality Factors: Events or seasons affecting sales, such as holidays or festivals. Example values are Festival, Holiday, NaN]
            [External Factors: Market conditions affecting sales, such as competitor pricing, economic indicators, or weather conditions. Example values are Competitor Pricing, Weather, Economic Indicator, NaN]
            [Demand Trend: Describes the sales trend for the product over time, whether it is increasing, stable, or decreasing. Example values: Increasing, Stable, Decreasing]
            [Customer Segments: The target customer group for the product, such as Regular, Premium, or Budget. Example values: Regular, Premium, Budget]""",
        ),
    ),
    QueryEngineTool(
        query_engine=inventory_audits_query_engine,
        metadata=ToolMetadata(
            name="inventary_and_supply_chain_data",
            description="""
            This dataset provides inventory and supply chain information for individual products across stores. 
            It includes - 
            Product ID: Unique identifier for each product. Example values are 9286, 2605, 2859.
            Store ID: Unique identifier for the store where the product is stocked. Example values are 16, 60, 55.
            Stock Levels: The current inventory level of the product in the store. Example values are 700, 82, 145.
            Supplier Lead Time (days): The number of days required for suppliers to deliver restocked products. Example values are 10, 11, 25.
            Stockout Frequency: The frequency at which the product runs out of stock. Example values are 14, 1, 14.
            Reorder Point: The stock level at which a new order should be placed to prevent stockouts. Example values are 132, 127, 192.
            Expiry Date: The expiration date of the product in YYYY-MM-DD format. Example values are 2024-01-15, 2024-12-16, 2024-04-30.
            Warehouse Capacity: The storage capacity of the warehouse where the product is kept. Example values are 1052, 1262, 1457.
            Order Fulfillment Time (days): The number of days required to fulfill an order after placement. Example values are 6, 9, 12.
        """),
    ),
    QueryEngineTool(
        query_engine=pricing_and_market_performance_query_engine,
        metadata=ToolMetadata(
            name="pricing_and_market_performance_query_data",
            description="""This dataset provides insights into product pricing, market competitiveness, and customer response. 
            It includes Product ID, Store ID, Price, Competitor Prices, Discounts, Sales Volume, Customer Reviews, Return Rate,
            Storage Cost, and Elasticity Index. This data is useful for analyzing price competitiveness, discount effectiveness, 
            customer satisfaction, and demand elasticity..
            It includes 
           [Product ID: Unique identifier for each product. Example values are 9502, 2068, 7103], 
           [Store ID: Unique identifier for the store selling the product. Example values are 13, 77, 59], 
           [Price: The selling price of the product per unit. Example values are 31.61, 35.51, 6.54],
           [Competitor Prices: The price of the same product sold by competitors. Example values are 56.14, 63.04, 30.61.],
           [Discounts: The percentage or absolute discount applied to the product. Example values are 19.68, 16.88, 10.86],
           [Sales Volume: The number of units sold. Example values are 255, 5, 184],
           [Customer Reviews: The average rating given by customers. Example values are 3, 3, 3],
           [Return Rate (%): The percentage of units returned by customers. Example values are 13.33, 1.50, 9.44],
           [Storage Cost: The cost of storing the product in a warehouse. Example values are 6.72, 8.38, 3.86],
           [Elasticity Index: A measure of how demand responds to changes in price. Example values are 1.78, 1.67, 2.46]           
            """,
        ),
    ),
]


from interactive1 import initialize_interactive_display
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

agent1 = ReActAgent.from_tools(tools, llm=llm1, verbose=True)
agent2 = ReActAgent.from_tools(tools, llm=llm2, verbose=True)
agent3 = ReActAgent.from_tools([], llm=llm1, verbose=True)

# Initialize session state for agent thinking
if "agent_thinking1" not in st.session_state:
    st.session_state["agent_thinking1"] = []
if "agent_thinking2" not in st.session_state:
    st.session_state["agent_thinking2"] = []
if "agent_thinking3" not in st.session_state:
    st.session_state["agent_thinking3"] = []
    
if "current_query_index" not in st.session_state:
    st.session_state["current_query_index"] = 0
    st.session_state["query_thinking_ranges"] = []
else:
    # Store the end index of the previous query
    if st.session_state["current_query_index"] < len(st.session_state["query_thinking_ranges"]):
        prev_end = len(st.session_state["agent_thinking1"])
        st.session_state["query_thinking_ranges"][st.session_state["current_query_index"]] = (
            st.session_state["query_thinking_ranges"][st.session_state["current_query_index"]][0],
            prev_end
        )
    
    # Start a new query
    st.session_state["current_query_index"] += 1
    start_idx = len(st.session_state["agent_thinking1"])
    st.session_state["query_thinking_ranges"].append((start_idx, start_idx))
    
st.markdown("""
    <style>
        .block-container {
            padding-bottom: 0px !important;
            # background: radial-gradient(circle at top left, #5680e9, #84ceeb, #5ab9ea, #8860d0) !important;
            # background: white !important;
            # color: white !important;
        }
        h1 {
            font-size: 28px !important;
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
    st.markdown("""
    <h1 style='font-size: 6px;'>Accenture Research Agent</h1>
    """, unsafe_allow_html=True)

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    chat_container = st.container(height=300, border=True)
    
    # Display all messages inside the fixed-height container
    with chat_container:
        st.info("""
        Welcome to Accenture Research Agent!  
        Feel free to query me regarding any questions. I am a Multi AI agent developed by Nishant from NIT SURAT, who is particpating in Accenture AI 
        Hackathon. I would be glad to help you. I have knowledge of the following datasets.
        """)
        
        # Display dataframe previews in expandable sections
        with st.expander("Available Datasets", expanded=False):
            tabs = st.tabs(["Products", "Inventory Audits", "Pricing & Market Data"])
            
            with tabs[0]:
                st.subheader("Products Dataset")
                st.dataframe(specific_products_df.head(3), use_container_width=True)
            
            with tabs[1]:
                st.subheader("Inventory Audits Dataset")
                st.dataframe(inventory_audits.head(3), use_container_width=True)
            
            with tabs[2]:
                st.subheader("Pricing & Market Data Dataset")
                st.dataframe(pricing_and_market_data.head(3), use_container_width=True)
        for message in st.session_state["messages"]:
            if message["role"] == "assistant":
                with st.chat_message(message["role"], avatar="Accenture-logo.png"):  # Chatbot icon
                    st.markdown(message["content"])
            else:  # user
                with st.chat_message(message["role"], avatar="9.png"):  # Person icon
                    st.markdown(message["content"])

    
    # User input outside the scrolling chat container
    if prompt := st.chat_input("What can I help you with?"):
        # Adding user message to session state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            # Initialized responses and thinking for all agents
            response1 = None
            response2 = None
            response3 = None

            
            # AGENT 1
            try:
                # A buffer to capture stdout
                buffer1 = io.StringIO()
                
                # Redirected stdout to our buffer and ran the agent
                with redirect_stdout(buffer1):
                    response1 = agent1.query(prompt)
                
                # Got the captured output and logged it
                captured_output1 = buffer1.getvalue()
                structured_steps1 = parse_agent_output(captured_output1)
                st.session_state["agent_thinking1"] = structured_steps1.splitlines()
            except Exception as e:
                st.session_state["agent_thinking1"] = [f"Agent 1 encountered an error: {str(e)}"]
                response1 = "Failed to process"

            # AGENT 2
            try:
                # Create a buffer to capture stdout
                buffer2 = io.StringIO()
                # Redirect stdout to our buffer and run the agent
                with redirect_stdout(buffer2):
                    if response1 == "Failed to process":
                        # Use response2 if available, otherwise use the original prompt
                        response2 = agent2.query(
                            f"""
                            Provide a comprehensive answer to the user's query: "{prompt}"
                            Use required tools from the set of tools provided to you.
                            """
                        )
                        # Get the captured output and log it
                        captured_output2 = buffer2.getvalue()
                        structured_steps2 = parse_agent_output(captured_output2)
                        st.session_state["agent_thinking2"] = structured_steps2.splitlines()
                    else:
                        response2 = "Agent 1 has already performed the task"
                        st.session_state["agent_thinking2"] = "Agent 1 has already performed the task"
                
            except Exception as e:
                st.session_state["agent_thinking2"] = [f"Agent 2 encountered an error: {str(e)}"]
                response2 = "Failed to process"

            # AGENT 3
            try:
                # Create a buffer to capture stdout
                buffer3 = io.StringIO()
                # Redirect stdout to our buffer and run the agent
                with redirect_stdout(buffer3):
                    response3 = agent3.query(
                        f"""
                        Agent 1 Thinking:
                        {' '.join(st.session_state["agent_thinking1"])}
                        
                        Agent 2 Thinking:
                        {' '.join(st.session_state["agent_thinking2"])}
                        
                        Based on these analysis results (note that some agents may have failed), 
                        provide a comprehensive answer to the user's query: "{prompt}"
                        Use whatever information is available from the successful agents.
                        """
                    )
                
                # Get the captured output and log it
                captured_output3 = buffer3.getvalue()
                structured_steps3 = parse_agent_output(captured_output3)
                st.session_state["agent_thinking3"] = structured_steps3.splitlines()
            except Exception as e:
                st.session_state["agent_thinking3"] = [f"Agent 3 encountered an error: {str(e)}"]
                response3 = "I apologize, but I'm having trouble processing your request right now. Could you please try again?"

            # Use response3 if available, otherwise provide a fallback message
            if response3 and response3 != "Failed to process":
                final_response = response3 
            elif response1:
                final_response = response1
            else:
                final_response = "I apologize, but I'm having trouble processing your request right now. Could you please try again?"
        
        # Add assistant response to session state
        st.session_state["messages"].append({"role": "assistant", "content": final_response})

        # Reset input box to default after response is received
        st.session_state["waiting_for_response"] = False
        st.session_state["last_prompt"] = "What can I help you with?"
        # Force rerun to update the chat history
        st.rerun()


# Right column - Agent Thinking Process 
with right_col:
    initialize_interactive_display()
    

# Footer
st.markdown("---")
st.markdown("© 2025 Accenture Retail Analytics Agent | Built by - Nishant - NIT SURAT | Email - nishant0363@gmail.com | Contact - 9306145426 ")
