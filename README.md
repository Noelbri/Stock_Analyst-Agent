# Stock Analyst Agent

This project is a financial advisory system where multiple agents collaborate to provide comprehensive stock analysis. The system is designed to handle user queries by gathering and analyzing market data, performing financial analysis, and retrieving news updates.

## Objective

The primary objective of this system is to create a multi-agent framework that streamlines the process of financial inquiry and analysis. By delegating tasks to specialized agents, the system can provide a cohesive and structured response to a wide range of client needs, from real-time data retrieval to in-depth financial analysis, and news aggregation.

### How it Works

The system is composed of several specialized agents and a supervisor agent that orchestrates the workflow:

*   **Market Data Agent:** Retrieves real-time stock prices and relevant financial metrics using the `yfinance` library. This agent provides clients with direct insights into the current market conditions for specified stocks or assets.
*   **Financial Analysis Agent:** Processes financial data to compute metrics such as return on investment (ROI) and growth rates. This is useful for clients seeking evaluative insights about potential or existing investments.
*   **Financial News Agent:** Pulls relevant financial news articles and updates, offering summarized insights on market trends, company announcements, and economic developments.
*   **Supervisor Agent:** Acts as an orchestrator, delegating the client's query to the appropriate agent based on the nature of the request. If the query is for stock prices or basic market information, it is routed to the Market Data Agent. For deeper analysis, such as calculating financial performance or finding relevant market news, the supervisor routes the query to the Financial Analysis Agent or the Financial News Agent.

This multi-agent approach ensures a streamlined and efficient workflow, where each agent addresses a distinct aspect of the financial query, ultimately creating a comprehensive and structured response.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/username/stock-analyst-agent.git
    cd stock-analyst-agent
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    GROQ_API_KEY="YOUR_GROQ_API_KEY"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    GROQ_MODEL="llama-3.3-70b-versatile"
    ```

5.  **Run the agent:**
    ```bash
    python3 agent.py
    ```

## Refactoring

The codebase has been refactored to improve readability, maintainability, and configuration. The following changes were made:

*   **Dependency Management:** A `requirements.txt` file has been added to manage project dependencies.
*   **Configuration:** The `GROQ_MODEL` is now configurable through the `.env` file.
*   **Code Readability:** Variable names have been made more descriptive, and comments and docstrings have been added to the code.
*   **Error Handling:** The code has been validated to ensure it runs without errors.
*   **Code Structure:** Placeholders have been added for the missing agent and graph definitions to guide further development.
