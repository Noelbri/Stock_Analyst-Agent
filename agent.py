import os 
from dotenv import load_dotenv
from functools import partial
from typing import Annotated, Sequence, TypedDict, Literal
import yfinance as yf
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
import functools
import operator

load_dotenv()

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#LLM definition
llm = ChatGroq( model="llama-3.3-70b-versatile", apikey=GROQ_API_KEY)

# Route Response structure for supervisor's decision
class RouteResponseFin(BaseModel):
    next: Literal["Market_Data_Agent", "Analysis_Agent", "News_Agent", "FINISH"]

# Define agent members
members_fin = ["Market_Data_Agent", "Analysis_Agent", "News_Agent"]

# Supervisor prompt setup
system_prompt_fin = (
    "You are a Financial Service Supervisor managing the following agents:"
    f"{','.join(members_fin)}. Select the next agent to handle the current query."
)

prompt_fin = ChatPromptTemplate.from_messages([
    ("system", system_prompt_fin),
    MessagePlaceholder(variable_name="messages"),
    ("system", "Choose the next agent from: {options}."),
]).partial(options=str(members_fin))

# Supervisor agent
def supervisor_agent_fin(state):
    supervisor_chain_fin = prompt_fin | llm.with_structured_output(RouteResponseFin)
    return supervisor_chain_fin.invoke(state)

# Define tools and Agent Prompts
# Market Data tool and Agent prompt
def fetch_stock_price(query):
    """Fetch the current stock price of a given stock symbol."""
    stock_symbol = query.split()[-1]
    stock = yf.Ticker(stock_symbol)
    try:
        current_price = stock.info.get("currentPrice")
        return f"The current stock price of {stock_symbol} is ${current_price}."
    except Exception as e:
        return f"Error retrieving stock data for {stock_symbol}: {str(e)}"
    
def agent_node(state, agent, name):
    result = agent.invoke(state)
    print(f"{name} Output: {result['messages'][-1].content}")
    return {
        "messages": [HumanMessage(content=result['messages'][-1].content, name=name)]
    }

market_data_prompt = (
    "You are the Market Data Agent. Your role is to retrieve the latest stock prices or"
    "market information based on user queries. Ensure your response includes the current price"
    "and any relevant market details if available."
)
market_data_agent = create_react_agent(llm, tools=[fetch_stock_price], state_modifier=market_data_prompt)
market_data_node = functools.partial(agent_node, agent=market_data_agent, name="Market_Data_Agent")

# Financial Analysis Tool and Agent prompt
def perform_financial_analysis(query):
    """Perform financial analysis based on user query."""
    if "ROI" in query:
        initial_investment = 1000
        final_value = 1200
        roi = (final_value - initial_investment) / initial_investment * 100
        return f"for an initial investment of ${initial_investment} yeilding ${final_value}, the ROI is{roi}%."
    return "No relevant financial analysis found."