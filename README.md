# Stock Analyst Agent

This project is an advanced financial advisory system where multiple specialized agents collaborate to provide comprehensive stock analysis, cryptocurrency data, and financial insights. The system is designed to handle complex user queries by intelligently routing them to the appropriate agent for processing and analysis.

## Objective

The primary objective of this system is to create a robust multi-agent framework that streamlines the process of financial inquiry and analysis. By delegating tasks to specialized agents, the system provides cohesive and structured responses to a wide range of financial needs, from real-time data retrieval to in-depth financial analysis and news aggregation.

## Key Features

### Enhanced Market Data Capabilities
- Real-time stock price retrieval for major exchanges
- Cryptocurrency price tracking (Bitcoin, Ethereum, Litecoin, Dogecoin)
- Comprehensive stock information including market cap, P/E ratios, 52-week highs/lows
- Volume and trading data analysis
- Multi-source price validation with fallback mechanisms

### Advanced Financial Analysis
- Dynamic ROI calculations using real historical data
- Growth rate computations with actual market performance
- P/E ratio analysis and interpretation
- Financial metrics comparison across time periods
- Custom calculation formulas for investment planning

### Financial News Intelligence
- Real-time financial news aggregation using Tavily Search
- Market trend analysis and summarization
- Company-specific news filtering
- Economic development tracking
- News sentiment analysis for investment decisions

### Intelligent Query Routing
- Advanced supervisor agent for optimal task delegation
- Context-aware routing based on query complexity
- Efficient workflow management to prevent redundant processing
- Smart termination conditions to avoid recursive loops

## How it Works

The system is composed of several specialized agents orchestrated by a supervisor agent:

### Supervisor Agent
Acts as the central orchestrator, analyzing incoming queries and routing them to the most appropriate specialized agent. The supervisor uses advanced natural language understanding to determine query intent and select the optimal processing path.

### Market Data Agent
- **Primary Function**: Retrieves real-time stock prices and market data
- **Data Sources**: Yahoo Finance API integration
- **Capabilities**: 
  - Stock price retrieval for major exchanges
  - Cryptocurrency price tracking with automatic symbol mapping
  - Comprehensive financial metrics (market cap, volume, trading ranges)
  - Error handling and data validation
- **Supported Assets**: 
  - Stocks: AAPL, GOOGL, MSFT, TSLA, AMZN, and thousands more
  - Cryptocurrencies: Bitcoin (BTC), Ethereum (ETH), Litecoin (LTC), Dogecoin (DOGE)

### Financial Analysis Agent
- **Primary Function**: Performs complex financial calculations and analysis
- **Capabilities**:
  - Real-time ROI calculations using historical market data
  - Growth rate analysis across multiple time periods
  - Financial ratio computations and interpretations
  - Custom investment scenario modeling
- **Analysis Types**:
  - Return on Investment (ROI) with historical data integration
  - Price-to-Earnings (P/E) ratio analysis
  - Growth rate calculations
  - Investment performance metrics

### Financial News Agent
- **Primary Function**: Aggregates and summarizes financial news
- **Data Sources**: Tavily Search API for real-time news
- **Capabilities**:
  - Real-time financial news retrieval
  - Market trend identification
  - Company-specific news filtering
  - Economic indicator tracking
  - News summarization without source attribution

## Technical Implementation

### Architecture
- **Framework**: LangChain with LangGraph for workflow orchestration
- **LLM Integration**: Groq API with Llama 3.3 70B model
- **Data Sources**: Yahoo Finance (yfinance), Tavily Search API
- **State Management**: TypedDict-based state handling with message persistence

### Key Improvements
- **Error Handling**: Comprehensive exception handling across all agents
- **Response Optimization**: Duplicate response elimination and content filtering
- **Recursion Management**: Configurable recursion limits to prevent infinite loops
- **API Integration**: Updated to latest LangChain Tavily package for improved performance

### Configuration Requirements
- **Environment Variables**: GROQ_API_KEY, TAVILY_API_KEY
- **Dependencies**: All packages specified in virtual environment
- **Python Version**: Compatible with Python 3.8+

## Usage Examples

### Interactive Mode
```bash
python3 agent.py
```
This launches the interactive chat interface where users can ask natural language questions about stocks, cryptocurrencies, and financial analysis.

### Programmatic Usage
```python
from agent import query_stock_analyst

# Get stock price
result = query_stock_analyst("What is the current price of Apple stock?")

# Get cryptocurrency data
result = query_stock_analyst("What is the current price of Bitcoin?")

# Perform financial analysis
result = query_stock_analyst("What is the ROI for Apple stocks over the past year?")

# Get financial news
result = query_stock_analyst("What are the latest news about Tesla?")
```

## Supported Query Types

### Market Data Queries
- "What is the current price of [STOCK_SYMBOL]?"
- "Show me information about [COMPANY_NAME] stock"
- "What is the current Bitcoin price?"
- "Get me the market cap of [STOCK_SYMBOL]"

### Financial Analysis Queries
- "What is the ROI for [STOCK_SYMBOL] over the past year?"
- "Calculate the growth rate for [STOCK_SYMBOL]"
- "What is the P/E ratio of [STOCK_SYMBOL]?"
- "Analyze the financial performance of [STOCK_SYMBOL]"

### Financial News Queries
- "What are the latest news about [COMPANY_NAME]?"
- "Show me recent financial market trends"
- "What are the current economic developments?"
- "Get me news about [SECTOR] industry"

## Installation and Setup

1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
6. Run the agent: `python3 agent.py`

## Project Structure

```
Stock_Analyst-Agent/
├── agent.py              # Main application file with all agent implementations
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (API keys)
├── .gitignore          # Git ignore file
├── README.md           # This file
└── venv/               # Virtual environment directory
```

## Performance Features

- **Response Time Optimization**: Intelligent caching and efficient API calls
- **Error Recovery**: Graceful handling of API failures and data inconsistencies
- **Resource Management**: Configurable recursion limits and timeout handling
- **Data Validation**: Comprehensive input validation and sanitization

This multi-agent approach ensures a streamlined and efficient workflow, where each agent addresses distinct aspects of financial queries, ultimately creating comprehensive and accurate responses for users seeking financial information and analysis.
