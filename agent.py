import os 
from dotenv import load_dotenv
from functools import partial
from typing import Annotated, Sequence, TypedDict, Literal
import yfinance as yf
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# Route Response structure for supervisor's decision
class RouteResponseFin(BaseModel):
    next: Literal["Market_Data_Agent", "Analysis_Agent", "News_Agent", "FINISH"]

# Define agent members
members_fin = ["Market_Data_Agent", "Analysis_Agent", "News_Agent"]

# Supervisor prompt setup
system_prompt_fin = (
    "You are a Financial Service Supervisor managing the following agents: "
    f"{','.join(members_fin)}. Given the conversation history, select the MOST APPROPRIATE agent to handle the current query. "
    "IMPORTANT: After an agent provides a response, you should typically choose 'FINISH' unless additional information is needed.\n\n"
    "Choose based on the query type:\n"
    "- Market_Data_Agent: for stock prices, cryptocurrency prices, market data, stock information\n"
    "- Analysis_Agent: for financial calculations, ROI, growth rates, financial analysis, P/E ratios\n"
    "- News_Agent: for financial news, market trends, current events, company news\n"
    "- FINISH: when the query has been adequately addressed by an agent\n\n"
    "Avoid sending the same query to multiple agents. Choose the most relevant one."
)

prompt_fin = ChatPromptTemplate.from_messages([
    ("system", system_prompt_fin),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Choose the next agent from: {options}."),
]).partial(options=str(members_fin))

# Supervisor agent
def supervisor_agent_fin(state):
    supervisor_chain_fin = prompt_fin | llm.with_structured_output(RouteResponseFin)
    result = supervisor_chain_fin.invoke(state)
    return {"next": result.next}

# Define tools and Agent Prompts
# Market Data tool and Agent prompt
def fetch_stock_price(query):
    """Fetch the current stock price of a given stock symbol or cryptocurrency."""
    # Extract symbol from query
    words = query.split()
    stock_symbol = words[-1].upper()
    
    # Handle common cryptocurrency symbols
    crypto_map = {
        "BITCOIN": "BTC-USD",
        "BTC": "BTC-USD",
        "ETHEREUM": "ETH-USD",
        "ETH": "ETH-USD",
        "LITECOIN": "LTC-USD",
        "LTC": "LTC-USD",
        "DOGECOIN": "DOGE-USD",
        "DOGE": "DOGE-USD"
    }
    
    if stock_symbol in crypto_map:
        stock_symbol = crypto_map[stock_symbol]
    elif any(crypto in query.lower() for crypto in ["bitcoin", "btc", "ethereum", "eth", "crypto"]):
        if "bitcoin" in query.lower() or "btc" in query.lower():
            stock_symbol = "BTC-USD"
        elif "ethereum" in query.lower() or "eth" in query.lower():
            stock_symbol = "ETH-USD"
    
    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        # Try different price fields
        current_price = (info.get("currentPrice") or 
                        info.get("regularMarketPrice") or 
                        info.get("previousClose"))
        
        if current_price:
            asset_type = "cryptocurrency" if "-USD" in stock_symbol else "stock"
            return f"The current {asset_type} price of {stock_symbol} is ${current_price:.2f}."
        else:
            return f"Could not retrieve current price for {stock_symbol}. Please check the symbol."
            
    except Exception as e:
        return f"Error retrieving data for {stock_symbol}: {str(e)}"

def get_stock_info(query):
    """Get comprehensive stock information including financials."""
    words = query.split()
    stock_symbol = words[-1].upper()
    
    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        result = f"Stock Information for {stock_symbol}:\n"
        result += f"Company: {info.get('longName', 'N/A')}\n"
        result += f"Current Price: ${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}\n"
        result += f"Market Cap: ${info.get('marketCap', 'N/A'):,}\n" if info.get('marketCap') else "Market Cap: N/A\n"
        result += f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
        result += f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
        result += f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
        result += f"Volume: {info.get('volume', 'N/A'):,}\n" if info.get('volume') else "Volume: N/A\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving information for {stock_symbol}: {str(e)}"
    
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [HumanMessage(content=result['messages'][-1].content, name=name)]
    }

market_data_prompt = (
    "You are the Market Data Agent. Your role is to retrieve the latest stock prices, "
    "cryptocurrency prices, and market information based on user queries. "
    "Use the fetch_stock_price tool for price information and get_stock_info for comprehensive data. "
    "Always provide current and accurate market data."
)
market_data_agent = create_react_agent(llm, tools=[fetch_stock_price, get_stock_info])
market_data_node = functools.partial(agent_node, agent=market_data_agent, name="Market_Data_Agent")

# Financial Analysis Tool and Agent prompt
def perform_financial_analysis(query):
    """Perform financial analysis based on user query."""
    query_lower = query.lower()
    
    # Handle ROI calculations
    if "roi" in query_lower or "return on investment" in query_lower:
        # Check if specific stock is mentioned
        if "apple" in query_lower or "aapl" in query_lower:
            try:
                stock = yf.Ticker("AAPL")
                hist = stock.history(period="1y")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    year_ago_price = hist['Close'].iloc[0]
                    roi = ((current_price - year_ago_price) / year_ago_price) * 100
                    return f"Apple (AAPL) 1-year ROI: {roi:.2f}%. Current price: ${current_price:.2f}, Price 1 year ago: ${year_ago_price:.2f}"
                else:
                    return "Unable to retrieve historical data for Apple stock."
            except Exception as e:
                return f"Error calculating Apple ROI: {str(e)}"
        else:
            # Generic ROI calculation example
            return ("ROI calculation formula: ROI = ((Final Value - Initial Value) / Initial Value) × 100%. "
                   "Please specify a stock symbol for detailed ROI analysis.")
    
    # Handle growth rate calculations
    elif "growth" in query_lower:
        return ("Growth rate calculation: ((New Value - Old Value) / Old Value) × 100%. "
               "Please provide specific values or stock symbol for calculation.")
    
    # Handle P/E ratio
    elif "p/e" in query_lower or "price to earnings" in query_lower:
        return ("P/E Ratio = Stock Price / Earnings Per Share. "
               "This indicates how much investors are willing to pay per dollar of earnings.")
    
    return "Please specify what type of financial analysis you need (ROI, growth rate, P/E ratio, etc.) and the stock symbol if applicable."

analysis_prompt = (
    "You are the financial analysis agent. Analyze the financial data provided in the query"
    "Perform calculations like ROI, growth rates, or other financial metrics as required."
    "provide a clear and concise response"
    "Only use the following tools:"
    "perform_financial_analysis"
)
analysis_agent = create_react_agent(llm, tools=[perform_financial_analysis])
analysis_node = functools.partial(agent_node, agent=analysis_agent, name="Analysis_Agent")

# Financial News Tool and Agent prompt
financial_news_tool = TavilySearch(max_results=5, api_key=TAVILY_API_KEY)
news_prompt = (
    "You are the financial news agent. Retrieve the latest financial news articles relevant to the user's query."
    "Use search tools to gather up-to-date news information and summarize key points."
    "Do not quote sources, just give a summary."
)
financial_news_agent = create_react_agent(llm, tools=[financial_news_tool])
news_node = functools.partial(agent_node, agent=financial_news_agent, name="News_Agent")

# Define Workflow state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Define the workflow with the supervisor and agent nodes
workflow_fin = StateGraph(AgentState)
workflow_fin.add_node("Market_Data_Agent", market_data_node)
workflow_fin.add_node("Analysis_Agent", analysis_node)
workflow_fin.add_node("News_Agent", news_node)
workflow_fin.add_node("Supervisor_Agent", supervisor_agent_fin)

# Define edges for agents to return to the supervisor
for member in members_fin:
    workflow_fin.add_edge(member, "Supervisor_Agent")

# Conditional map for routing based on supervisor's decision
conditional_map_fin = {
    "Market_Data_Agent": "Market_Data_Agent",
    "Analysis_Agent": "Analysis_Agent",
    "News_Agent": "News_Agent",
    "FINISH": END  # This will end the workflow when the supervisor decides
}
workflow_fin.add_conditional_edges("Supervisor_Agent", lambda x:x["next"], conditional_map_fin)
workflow_fin.add_edge(START, "Supervisor_Agent")

# Compile the workflow
graph_fin = workflow_fin.compile()

# Function to interact with the Stock Analyst Agent
def query_stock_analyst(user_query):
    """
    Function to query the Stock Analyst Agent with user input.
    
    Args:
        user_query (str): The user's question or request
    
    Returns:
        str: The agent's response
    """
    try:
        inputs = {"messages": [HumanMessage(content=user_query)]}
        
        # Collect all responses
        responses = []
        config = {"recursion_limit": 10}  # Set a reasonable recursion limit
        
        for output in graph_fin.stream(inputs, config=config):
            if "__end__" not in output:
                for key, value in output.items():
                    if key != "Supervisor_Agent" and "messages" in value and value["messages"]:
                        # Only collect unique responses from actual worker agents
                        response_content = value['messages'][-1].content
                        if response_content not in responses:
                            responses.append(response_content)
        
        # Return the most relevant response (usually the last one)
        return responses[-1] if responses else "No response generated."
        
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Interactive function for continuous user input
def run_interactive_agent():
    """
    Run the agent in interactive mode, allowing continuous user input.
    """
    print("=== Stock Analyst Agent ===")
    print("Available services:")
    print("- Market Data: Get current stock prices and cryptocurrency prices")
    print("- Comprehensive Stock Info: Detailed financial metrics and company data")
    print("- Financial Analysis: ROI calculations, growth rates, P/E ratios")
    print("- Financial News: Latest financial news summaries")
    print("\nSupported assets: Stocks (AAPL, GOOGL, etc.) and Cryptocurrencies (Bitcoin, Ethereum, etc.)")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    
    while True:
        try:
            user_input = input("Ask me anything about stocks or finance: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using Stock Analyst Agent!")
                break
            
            if not user_input:
                print("Please enter a valid question.")
                continue
            
            print("\nProcessing your request...")
            response = query_stock_analyst(user_input)
            print(f"\nResponse:\n{response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    # You can either run interactively or test with a single query
    
    # Single query example:
    # result = query_stock_analyst("What is the current stock price of AAPL?")
    # print(result)
    
    # Interactive mode:
    run_interactive_agent()