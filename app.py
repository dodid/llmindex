import os
import random
import string
from datetime import date

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langchain_groq import ChatGroq

proxy_url = os.getenv('GROQ_PROXY') or None


def generate_dummy_stock_symbols(count, length=4):
    """
    Generates a list of dummy stock symbols.

    Parameters:
    count (int): The number of stock symbols to generate.
    length (int): The length of each stock symbol. Defaults to 4.

    Returns:
    list: A list containing the generated stock symbols.
    """
    symbols = []
    for _ in range(count):
        symbol = ''.join(random.choices(string.ascii_uppercase, k=length))
        symbols.append(symbol)
    return symbols


def get_dummy_stock_info(symbol, industry=None, sector=None, country=None):
    """
    Returns dummy metadata information for a given stock symbol.

    Parameters:
    symbol (str): The stock symbol.
    industry (str, optional): The industry of the company. Defaults to None.
    sector (str, optional): The sector of the company. Defaults to None.
    country (str, optional): The country of the company. Defaults to None.

    Returns:
    dict: A dictionary containing metadata information about the stock.
    """
    industries = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary', 'Energy']
    sectors = ['Information Technology', 'Health Care', 'Finance', 'Consumer Services', 'Utilities']
    countries = ['United States', 'Canada', 'Germany', 'France', 'Japan']

    info = {
        'symbol': symbol,
        'companyName': f'{symbol} Inc.',
        'industry': industry if industry else random.choice(industries),
        'sector': sector if sector else random.choice(sectors),
        'country': country if country else random.choice(countries),
        'fullTimeEmployees': random.randint(1000, 100000),
        'marketCap': random.uniform(10e9, 500e9),
        'beta': random.uniform(0.5, 1.5),
        'dividendYield': random.uniform(0, 5),
        'priceToEarnings': random.uniform(10, 30),
        'forwardPE': random.uniform(10, 30),
        'priceToSales': random.uniform(1, 10),
        'priceToBook': random.uniform(1, 10),
        'revenue': random.uniform(1e9, 100e9),
        'grossProfits': random.uniform(1e8, 50e9),
        'freeCashFlow': random.uniform(1e8, 50e9),
        'ebitda': random.uniform(1e8, 50e9),
        'earningsGrowth': random.uniform(-0.2, 0.2),
        'revenueGrowth': random.uniform(-0.2, 0.2),
        'debtToEquity': random.uniform(0, 2),
        'currentRatio': random.uniform(0.5, 3),
        'quickRatio': random.uniform(0.5, 3),
        'totalAssets': random.uniform(1e9, 500e9),
        'totalLiabilities': random.uniform(1e9, 500e9),
        'operatingCashFlow': random.uniform(1e8, 50e9),
        'grossMargins': random.uniform(0.1, 0.9),
        'operatingMargins': random.uniform(0.1, 0.5),
        'profitMargins': random.uniform(0.1, 0.5),
        'sharesOutstanding': random.randint(10e6, 1e9),
        'sharesFloat': random.randint(10e6, 1e9),
        'sharesShort': random.randint(0, 10e6),
        'shortRatio': random.uniform(0, 5),
        'shortPercentOfFloat': random.uniform(0, 10),
        'shortPercentOfSharesOutstanding': random.uniform(0, 10),
        'recommendationKey': random.choice(['buy', 'outperform', 'hold', 'underperform', 'sell']),
        'recommendationMean': random.uniform(1, 5),
        'currentPrice': random.uniform(100, 500)
    }
    return info


def generate_random_walk(symbol, start_date, end_date, init_value=100, mean=0, stddev=0.01):
    """
    Generate a random walk time series for the closing prices of a stock.

    Parameters:
    symbol (str): The stock symbol (not used in computation, just for reference).
    start_date (str): The start date of the series (format: 'YYYY-MM-DD').
    end_date (str): The end date of the series (format: 'YYYY-MM-DD').
    init_value (float): The initial value of the stock price.
    mean (float): The mean of the daily returns.
    stddev (float): The standard deviation of the daily returns.

    Returns:
    pd.Series: The generated closing prices time series.
    """
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days

    # Number of days in the date range
    num_days = len(dates)

    # Generate random daily returns
    daily_returns = np.random.normal(loc=mean, scale=stddev, size=num_days)

    # Compute cumulative returns
    cumulative_returns = np.cumsum(daily_returns)

    # Compute price series
    price_series = init_value * np.exp(cumulative_returns)

    # Create a pandas Series
    price_series = pd.Series(data=price_series, index=dates, name=symbol)

    return price_series


@tool
def get_stock_info(symbol, key):
    '''Return the correct stock info value given the appropriate symbol and key. Infer valid key from the user prompt; it must be one of the following:

    symbol,companyName,industry,sector,country,fullTimeEmployees,marketCap,beta,dividendYield,priceToEarnings,forwardPE,priceToSales,priceToBook,revenue,grossProfits,freeCashFlow,ebitda,earningsGrowth,revenueGrowth,debtToEquity,currentRatio,quickRatio,totalAssets,totalLiabilities,operatingCashFlow,grossMargins,operatingMargins,profitMargins,sharesOutstanding,sharesFloat,sharesShort,shortRatio,shortPercentOfFloat,shortPercentOfSharesOutstanding,recommendationKey,recommendationMean,currentPrice

    If asked generically for 'stock price', use currentPrice
    '''
    data = get_dummy_stock_info(symbol)
    return data[key]


@tool
def get_historical_price(symbol, start_date, end_date):
    """
    Fetches historical stock prices for a given symbol from 'start_date' to 'end_date'.
    - symbol (str): Stock ticker symbol.
    - end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date
    - start_date (date): Set explicitly, or calculated as 'end_date - date interval' (for example, if prompted 'over the past 6 months', date interval = 6 months so start_date would be 6 months earlier than today's date). Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date
    """

    return generate_random_walk(symbol, start_date, end_date)


def plot_price_over_time(price_series):

    fig = plt.figure(figsize=(10, 6))
    for s in price_series:
        plt.plot(s.index, s, label=s.name)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Historical Stock Prices')
    plt.legend()
    st.pyplot(fig)


def call_functions(llm_with_tools, user_prompt):
    system_prompt = 'You are a helpful finance assistant that analyzes stocks and stock prices. Today is {today}'.format(
        today=date.today())

    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    historical_price_dfs = []
    symbols = []
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"get_stock_info": get_stock_info,
                         "get_historical_price": get_historical_price}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        if tool_call['name'] == 'get_historical_price':
            historical_price_dfs.append(tool_output)
            symbols.append(tool_output.name)
        else:
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    if len(historical_price_dfs) > 0:
        plot_price_over_time(historical_price_dfs)

        symbols = ' and '.join(symbols)
        messages.append(
            ToolMessage(
                'Tell the user that a historical stock price chart for {symbols} been generated.'.format(
                    symbols=symbols),
                tool_call_id=0))

    return llm_with_tools.invoke(messages).content


def help():

    st.write(generate_dummy_stock_symbols(5))

    multiline_text = """
    This application uses the Llama 3 model to answer questions about stocks and stock prices.

    You can ask questions like:
    - "What is the current price of Apple stock?"
    - "Show me the historical prices of Apple vs Microsoft stock over the past 6 months."

    **Note: The application uses dummy data for stock information and historical stock prices.**
    """

    st.sidebar.markdown(multiline_text, unsafe_allow_html=True)


def main():

    help()

    proxy_params = {'http_client': httpx.Client(
        proxies={"http://": proxy_url, "https://": proxy_url})} if proxy_url else {}

    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model='llama3-8b-8192',
        **proxy_params
    )

    tools = [get_stock_info, get_historical_price]
    llm_with_tools = llm.bind_tools(tools)

    # Display the title and introduction of the application
    st.title("Index Design with Llama 3")

    # Get the user's question
    user_question = st.text_input("Ask a question about a stock or multiple stocks:")

    if user_question:
        response = call_functions(llm_with_tools, user_question)
        st.write(response)


if __name__ == "__main__":
    main()
