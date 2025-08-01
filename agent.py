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
    "- Market_Data_Agent: for stock prices, cryptocurrency prices, market data, stock information, financial status queries, comprehensive company data\n"
    "- Analysis_Agent: for financial analysis, financial health assessment, performance evaluation, ROI calculations, growth rates, P/E ratios, investment analysis\n"
    "- News_Agent: for financial news, market trends, current events, company news, recent developments\n"
    "- FINISH: when the query has been adequately addressed by an agent\n\n"
    "For queries asking about 'financial status' or 'how is [company]', route to Market_Data_Agent first for comprehensive data, then Analysis_Agent if deeper analysis is needed.\n"
    "Avoid sending the same query to multiple agents unless additional perspective is specifically needed."
)

prompt_fin = ChatPromptTemplate.from_messages([
    ("system", system_prompt_fin),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Choose the next agent from: {options}."),
]).partial(options=str(members_fin))

# Supervisor agent
def supervisor_agent_fin(state):
    # Check if we have enough responses to finish
    messages = state.get("messages", [])
    agent_count = state.get("agent_count", 0)
    
    # If we've had 2 or more agent responses, finish to avoid loops
    if agent_count >= 2:
        return {"next": "FINISH"}
    
    # If we have responses from agents (not just user input), consider finishing
    agent_responses = [msg for msg in messages if hasattr(msg, 'name') and msg.name in members_fin]
    
    # If we already have a substantive response from an agent, finish
    if len(agent_responses) >= 1:
        # Check if the last agent response is substantive (not an error or "need more steps")
        last_response = agent_responses[-1].content if agent_responses else ""
        if (last_response and 
            len(last_response) > 50 and  # Substantive response
            "need more steps" not in last_response.lower() and
            "sorry" not in last_response.lower()[:20] and
            "error" not in last_response.lower()[:20]):  # Not starting with error
            return {"next": "FINISH"}
    
    # Otherwise, route to appropriate agent
    supervisor_chain_fin = prompt_fin | llm.with_structured_output(RouteResponseFin)
    result = supervisor_chain_fin.invoke(state)
    return {"next": result.next}

# Define tools and Agent Prompts
# Market Data tool and Agent prompt
def extract_stock_symbol(query):
    """Extract stock symbol from query text."""
    query_lower = query.lower()
    
    # Common company name to symbol mapping
    company_map = {
        "microsoft": "MSFT",
        "apple": "AAPL",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "meta": "META",
        "facebook": "META",
        "netflix": "NFLX",
        "nvidia": "NVDA",
        "intel": "INTC",
        "amd": "AMD",
        "salesforce": "CRM",
        "oracle": "ORCL",
        "ibm": "IBM",
        "cisco": "CSCO"
    }
    
    # Check for company names first
    for company, symbol in company_map.items():
        if company in query_lower:
            return symbol
    
    # Extract symbol from query (look for potential stock symbols)
    words = query.upper().split()
    for word in words:
        # Remove common punctuation
        clean_word = word.strip(".,!?:;")
        # Check if it looks like a stock symbol (2-5 letters, all caps)
        if len(clean_word) >= 2 and len(clean_word) <= 5 and clean_word.isalpha():
            return clean_word
    
    # Fallback to last word
    return words[-1].upper() if words else "UNKNOWN"

def fetch_stock_price(query):
    """Fetch the current stock price of a given stock symbol or cryptocurrency."""
    stock_symbol = extract_stock_symbol(query)
    
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
            company_name = info.get("longName", stock_symbol)
            return f"The current {asset_type} price of {company_name} ({stock_symbol}) is ${current_price:.2f}."
        else:
            return f"Could not retrieve current price for {stock_symbol}. Please check the symbol."
            
    except Exception as e:
        return f"Error retrieving data for {stock_symbol}: {str(e)}"

def get_stock_info(query):
    """Get comprehensive stock information including financials."""
    stock_symbol = extract_stock_symbol(query)
    
    try:
        stock = yf.Ticker(stock_symbol)
        info = stock.info
        
        # Get historical data for additional metrics
        hist = stock.history(period="1y")
        
        company_name = info.get('longName', stock_symbol)
        result = f"Financial Status for {company_name} ({stock_symbol}):\n\n"
        
        # Basic Information
        result += "=== CURRENT TRADING INFO ===\n"
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        result += f"Current Price: ${current_price}\n"
        result += f"Previous Close: ${info.get('previousClose', 'N/A')}\n"
        result += f"Day Change: {info.get('regularMarketChange', 'N/A')}\n"
        result += f"Day Change %: {info.get('regularMarketChangePercent', 'N/A'):.2f}%\n" if info.get('regularMarketChangePercent') else "Day Change %: N/A\n"
        
        # Market Metrics
        result += "\n=== MARKET METRICS ===\n"
        if info.get('marketCap'):
            result += f"Market Cap: ${info.get('marketCap'):,}\n"
        else:
            result += "Market Cap: N/A\n"
        result += f"Volume: {info.get('volume', 'N/A'):,}\n" if info.get('volume') else "Volume: N/A\n"
        result += f"Avg Volume: {info.get('averageVolume', 'N/A'):,}\n" if info.get('averageVolume') else "Avg Volume: N/A\n"
        
        # Price Ranges
        result += "\n=== PRICE RANGES ===\n"
        result += f"52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}\n"
        result += f"52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
        result += f"Day High: ${info.get('dayHigh', 'N/A')}\n"
        result += f"Day Low: ${info.get('dayLow', 'N/A')}\n"
        
        # Financial Ratios
        result += "\n=== FINANCIAL RATIOS ===\n"
        result += f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
        result += f"Forward P/E: {info.get('forwardPE', 'N/A')}\n"
        result += f"Price to Book: {info.get('priceToBook', 'N/A')}\n"
        result += f"Dividend Yield: {info.get('dividendYield', 'N/A'):.2f}%\n" if info.get('dividendYield') else "Dividend Yield: N/A\n"
        
        # Performance
        if not hist.empty and len(hist) > 0:
            result += "\n=== PERFORMANCE ===\n"
            year_return = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100) if isinstance(current_price, (int, float)) else 0
            result += f"1-Year Return: {year_return:.2f}%\n"
        
        # Financial Health
        result += "\n=== FINANCIAL HEALTH ===\n"
        result += f"Total Revenue: ${info.get('totalRevenue', 'N/A'):,}\n" if info.get('totalRevenue') else "Total Revenue: N/A\n"
        result += f"Gross Profit: ${info.get('grossProfits', 'N/A'):,}\n" if info.get('grossProfits') else "Gross Profit: N/A\n"
        result += f"Total Debt: ${info.get('totalDebt', 'N/A'):,}\n" if info.get('totalDebt') else "Total Debt: N/A\n"
        result += f"Total Cash: ${info.get('totalCash', 'N/A'):,}\n" if info.get('totalCash') else "Total Cash: N/A\n"
        
        return result
        
    except Exception as e:
        return f"Error retrieving comprehensive information for {stock_symbol}: {str(e)}"
    
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # Update agent count
    current_count = state.get("agent_count", 0)
    return {
        "messages": [HumanMessage(content=result['messages'][-1].content, name=name)],
        "agent_count": current_count + 1
    }

market_data_prompt = (
    "You are the Market Data Agent. Your role is to retrieve and provide comprehensive stock market information. "
    "For queries about 'financial status', 'stock information', or general company queries, use get_stock_info for detailed analysis. "
    "For simple price queries, use fetch_stock_price. "
    "Always provide current, accurate, and comprehensive market data. "
    "When a user asks about financial status, provide a detailed overview using get_stock_info."
)
market_data_agent = create_react_agent(llm, tools=[fetch_stock_price, get_stock_info])
market_data_node = functools.partial(agent_node, agent=market_data_agent, name="Market_Data_Agent")

# Financial Analysis Tool and Agent prompt
def perform_financial_analysis(query):
    """Perform financial analysis based on user query."""
    query_lower = query.lower()
    stock_symbol = extract_stock_symbol(query)
    
    # Handle financial status queries
    if any(term in query_lower for term in ["financial status", "financial health", "financial condition", "how is", "financial performance"]):
        try:
            stock = yf.Ticker(stock_symbol)
            info = stock.info
            hist = stock.history(period="1y")
            
            company_name = info.get('longName', stock_symbol)
            analysis = f"Financial Analysis for {company_name} ({stock_symbol}):\n\n"
            
            # Profitability Analysis
            analysis += "=== PROFITABILITY ANALYSIS ===\n"
            profit_margin = info.get('profitMargins')
            if profit_margin:
                analysis += f"Profit Margin: {profit_margin:.2%}\n"
                if profit_margin > 0.20:
                    analysis += "✓ Excellent profit margins (>20%)\n"
                elif profit_margin > 0.10:
                    analysis += "✓ Good profit margins (10-20%)\n"
                else:
                    analysis += "⚠ Lower profit margins (<10%)\n"
            
            # Growth Analysis
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth:
                analysis += f"Revenue Growth: {revenue_growth:.2%}\n"
                if revenue_growth > 0.15:
                    analysis += "✓ Strong revenue growth (>15%)\n"
                elif revenue_growth > 0.05:
                    analysis += "✓ Moderate revenue growth (5-15%)\n"
                else:
                    analysis += "⚠ Slow revenue growth (<5%)\n"
            
            # Valuation Analysis
            analysis += "\n=== VALUATION ANALYSIS ===\n"
            pe_ratio = info.get('trailingPE')
            if pe_ratio:
                analysis += f"P/E Ratio: {pe_ratio:.2f}\n"
                if pe_ratio < 15:
                    analysis += "✓ Potentially undervalued (P/E < 15)\n"
                elif pe_ratio < 25:
                    analysis += "✓ Reasonably valued (P/E 15-25)\n"
                else:
                    analysis += "⚠ Potentially overvalued (P/E > 25)\n"
            
            # Financial Strength
            analysis += "\n=== FINANCIAL STRENGTH ===\n"
            total_cash = info.get('totalCash', 0)
            total_debt = info.get('totalDebt', 0)
            if total_cash and total_debt:
                debt_to_cash = total_debt / total_cash if total_cash > 0 else float('inf')
                analysis += f"Debt-to-Cash Ratio: {debt_to_cash:.2f}\n"
                if debt_to_cash < 1:
                    analysis += "✓ Strong financial position (More cash than debt)\n"
                elif debt_to_cash < 2:
                    analysis += "✓ Reasonable financial position\n"
                else:
                    analysis += "⚠ High debt relative to cash\n"
            
            # Stock Performance
            if not hist.empty:
                analysis += "\n=== STOCK PERFORMANCE ===\n"
                current_price = info.get('currentPrice', info.get('regularMarketPrice'))
                if current_price and len(hist) > 0:
                    year_return = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    analysis += f"1-Year Return: {year_return:.2f}%\n"
                    if year_return > 20:
                        analysis += "✓ Excellent performance (>20% annual return)\n"
                    elif year_return > 10:
                        analysis += "✓ Good performance (10-20% annual return)\n"
                    elif year_return > 0:
                        analysis += "✓ Positive performance (0-10% annual return)\n"
                    else:
                        analysis += "⚠ Negative performance\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing {stock_symbol}: {str(e)}"
    
    # Handle ROI calculations
    elif "roi" in query_lower or "return on investment" in query_lower:
        try:
            stock = yf.Ticker(stock_symbol)
            hist = stock.history(period="1y")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                year_ago_price = hist['Close'].iloc[0]
                roi = ((current_price - year_ago_price) / year_ago_price) * 100
                company_name = stock.info.get('longName', stock_symbol)
                return f"{company_name} ({stock_symbol}) 1-year ROI: {roi:.2f}%. Current price: ${current_price:.2f}, Price 1 year ago: ${year_ago_price:.2f}"
            else:
                return f"Unable to retrieve historical data for {stock_symbol}."
        except Exception as e:
            return f"Error calculating ROI for {stock_symbol}: {str(e)}"
    
    # Handle growth rate calculations
    elif "growth" in query_lower:
        return ("Growth rate calculation: ((New Value - Old Value) / Old Value) × 100%. "
               "Please provide specific values or stock symbol for calculation.")
    
    # Handle P/E ratio
    elif "p/e" in query_lower or "price to earnings" in query_lower:
        return ("P/E Ratio = Stock Price / Earnings Per Share. "
               "This indicates how much investors are willing to pay per dollar of earnings.")
    
    return "Please specify what type of financial analysis you need (financial status, ROI, growth rate, P/E ratio, etc.) and the stock symbol if applicable."

analysis_prompt = (
    "You are the Financial Analysis Agent. Your expertise includes analyzing financial health, performance metrics, and investment potential. "
    "For queries about 'financial status', 'financial health', or 'how is [company]', use perform_financial_analysis to provide comprehensive analysis. "
    "You can perform calculations like ROI, growth rates, valuation analysis, and financial ratios. "
    "Always provide clear, actionable insights with supporting data and interpret what the numbers mean for investors."
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
    agent_count: int  # Track how many agents have responded

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
        inputs = {"messages": [HumanMessage(content=user_query)], "agent_count": 0}
        
        # Collect all responses
        responses = []
        config = {"recursion_limit": 10}  # Reduced recursion limit
        
        for output in graph_fin.stream(inputs, config=config):
            if "__end__" not in output:
                for key, value in output.items():
                    if key != "Supervisor_Agent" and "messages" in value and value["messages"]:
                        # Only collect unique responses from actual worker agents
                        response_content = value['messages'][-1].content
                        if response_content and response_content not in responses:
                            responses.append(response_content)
        
        # Return the most comprehensive response (usually the last one)
        if responses:
            return responses[-1]
        else:
            # Fallback: try direct market data agent for stock queries
            if any(term in user_query.lower() for term in ["stock", "microsoft", "apple", "financial status", "price", "analyze"]):
                try:
                    market_inputs = {"messages": [HumanMessage(content=user_query)]}
                    result = market_data_agent.invoke(market_inputs)
                    return result['messages'][-1].content if result.get('messages') else "Unable to process request."
                except Exception as fallback_error:
                    # Try analysis agent as final fallback
                    try:
                        analysis_inputs = {"messages": [HumanMessage(content=user_query)]}
                        result = analysis_agent.invoke(analysis_inputs)
                        return result['messages'][-1].content if result.get('messages') else "Unable to process request."
                    except:
                        pass
            return "I apologize, but I'm having trouble processing your request. Please try rephrasing your question or ask about specific stock symbols."
        
    except Exception as e:
        return f"Error processing query: {str(e)}. Please try again with a more specific question."

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