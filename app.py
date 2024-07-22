import functools
import os
import random
import string
from datetime import date
from typing import List, Optional

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from langchain.globals import set_debug
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langchain_groq import ChatGroq

set_debug(True)

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

    currentPrice,symbol,companyName,industry,sector,country,fullTimeEmployees,marketCap,beta,dividendYield,priceToEarnings,forwardPE,priceToSales,priceToBook,revenue,grossProfits,freeCashFlow,ebitda,earningsGrowth,revenueGrowth,debtToEquity,currentRatio,quickRatio,totalAssets,totalLiabilities,operatingCashFlow,grossMargins,operatingMargins,profitMargins,sharesOutstanding,sharesFloat,sharesShort,shortRatio,shortPercentOfFloat,shortPercentOfSharesOutstanding,recommendationKey,recommendationMean

    If asked generically for 'stock price', use currentPrice
    '''
    try:
        return st.session_state.symbol_universe.loc[symbol, key]
    except KeyError:
        return 'Not found'


@tool
def get_historical_price(symbol, start_date, end_date):
    """
    Fetches historical stock prices for a given symbol from 'start_date' to 'end_date'.
    - symbol (str): Stock ticker symbol.
    - end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date
    - start_date (date): Set explicitly, or calculated as 'end_date - date interval' (for example, if prompted 'over the past 6 months', date interval = 6 months so start_date would be 6 months earlier than today's date). Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date
    """

    return generate_random_walk(symbol, start_date, end_date)

@tool
def rank_stocks_by_metric(universe, metric, limit, ascending=True):
    """
    Ranks the stocks in the universe based on the provided metric. 

    metric can be currentPrice,fullTimeEmployees,marketCap,beta,dividendYield,priceToEarnings,forwardPE,priceToSales,priceToBook,revenue,grossProfits,freeCashFlow,ebitda,earningsGrowth,revenueGrowth,debtToEquity,currentRatio,quickRatio,totalAssets,totalLiabilities,operatingCashFlow,grossMargins,operatingMargins,profitMargins,sharesOutstanding,sharesFloat,sharesShort,shortRatio,shortPercentOfFloat,shortPercentOfSharesOutstanding,recommendationKey,recommendationMean

    - universe (str): The stock symbols to rank as comma separated list. Use None for the whole universe.
    - metric (str): The metric to rank the stocks by.
    - limit (int): The number of stocks to return.
    - ascending (bool): Whether to rank the stocks in ascending order. Defaults to True.
    """
    if universe in [None, 'all', '']:
        universe = st.session_state.symbol_universe.index.tolist()
    else:
        universe = universe.split(',')
    result = st.session_state.symbol_universe.loc[universe][metric].sort_values(ascending=ascending).index[:limit].tolist()
    return f'The stocks are ranked by {metric} in {"ascending" if ascending else "descending"} order: {", ".join(result)}'


@tool
def filter_stocks(universe=None,
                currentPriceMin=None, currentPriceMax=None,
                industry=None, sector=None, country=None,
                fullTimeEmployeesMin=None, fullTimeEmployeesMax=None,
                marketCapMin=None, marketCapMax=None,
                betaMin=None, betaMax=None,
                dividendYieldMin=None, dividendYieldMax=None,
                priceToEarningsMin=None, priceToEarningsMax=None,
                forwardPEMin=None, forwardPEMax=None,
                priceToSalesMin=None, priceToSalesMax=None,
                priceToBookMin=None, priceToBookMax=None,
                revenueMin=None, revenueMax=None,
                grossProfitsMin=None, grossProfitsMax=None,
                freeCashFlowMin=None, freeCashFlowMax=None,
                ebitdaMin=None, ebitdaMax=None,
                earningsGrowthMin=None, earningsGrowthMax=None,
                revenueGrowthMin=None, revenueGrowthMax=None,
                debtToEquityMin=None, debtToEquityMax=None,
                currentRatioMin=None, currentRatioMax=None,
                quickRatioMin=None, quickRatioMax=None,
                totalAssetsMin=None, totalAssetsMax=None,
                totalLiabilitiesMin=None, totalLiabilitiesMax=None,
                operatingCashFlowMin=None, operatingCashFlowMax=None,
                grossMarginsMin=None, grossMarginsMax=None,
                operatingMarginsMin=None, operatingMarginsMax=None,
                profitMarginsMin=None, profitMarginsMax=None,
                sharesOutstandingMin=None, sharesOutstandingMax=None,
                sharesFloatMin=None, sharesFloatMax=None,
                sharesShortMin=None, sharesShortMax=None,
                shortRatioMin=None, shortRatioMax=None,
                shortPercentOfFloatMin=None, shortPercentOfFloatMax=None,
                shortPercentOfSharesOutstandingMin=None, shortPercentOfSharesOutstandingMax=None,
                recommendationKey=None,
                recommendationMeanMin=None, recommendationMeanMax=None):
    """
    Filters the stock universe based on the provided criteria. 
    If an argument is not provided, it is not used as a filter.
    Do not invent values for unused arguments. 

    industries = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary', 'Energy']
    sectors = ['Information Technology', 'Health Care', 'Finance', 'Consumer Services', 'Utilities']
    countries = ['United States', 'Canada', 'Germany', 'France', 'Japan']

    Parameters:
        universe (str): The stock universe as comma separated values, the whole universe if None.
        currentPriceMin (float): The minimum current price of the stock.
        currentPriceMax (float): The maximum current price of the stock.
        industry (str): The industry of the company.
        sector (str): The sector of the company.
        country (str): The country of the company.  
        fullTimeEmployeesMin (int): The minimum number of full-time employees.
        fullTimeEmployeesMax (int): The maximum number of full-time employees.
        marketCapMin (float): The minimum market capitalization.
        marketCapMax (float): The maximum market capitalization.
        betaMin (float): The minimum beta value.
        betaMax (float): The maximum beta value.
        dividendYieldMin (float): The minimum dividend yield.
        dividendYieldMax (float): The maximum dividend yield.
        priceToEarningsMin (float): The minimum price-to-earnings ratio.
        priceToEarningsMax (float): The maximum price-to-earnings ratio.
        forwardPEMin (float): The minimum forward price-to-earnings ratio.
        forwardPEMax (float): The maximum forward price-to-earnings ratio.
        priceToSalesMin (float): The minimum price-to-sales ratio.
        priceToSalesMax (float): The maximum price-to-sales ratio.
        priceToBookMin (float): The minimum price-to-book ratio.
        priceToBookMax (float): The maximum price-to-book ratio.
        revenueMin (float): The minimum revenue.
        revenueMax (float): The maximum revenue.
        grossProfitsMin (float): The minimum gross profits.
        grossProfitsMax (float): The maximum gross profits.
        freeCashFlowMin (float): The minimum free cash flow.
        freeCashFlowMax (float): The maximum free cash flow.
        ebitdaMin (float): The minimum EBITDA.
        ebitdaMax (float): The maximum EBITDA.
        earningsGrowthMin (float): The minimum earnings growth.
        earningsGrowthMax (float): The maximum earnings growth.
        revenueGrowthMin (float): The minimum revenue growth.
        revenueGrowthMax (float): The maximum revenue growth.
        debtToEquityMin (float): The minimum debt-to-equity ratio.
        debtToEquityMax (float): The maximum debt-to-equity ratio.
        currentRatioMin (float): The minimum current ratio.
        currentRatioMax (float): The maximum current ratio.
        quickRatioMin (float): The minimum quick ratio.
        quickRatioMax (float): The maximum quick ratio.
        totalAssetsMin (float): The minimum total assets.
        totalAssetsMax (float): The maximum total assets.
        totalLiabilitiesMin (float): The minimum total liabilities.
        totalLiabilitiesMax (float): The maximum total liabilities.
        operatingCashFlowMin (float): The minimum operating cash flow.
        operatingCashFlowMax (float): The maximum operating cash flow.
        grossMarginsMin (float): The minimum gross margins.
        grossMarginsMax (float): The maximum gross margins.
        operatingMarginsMin (float): The minimum operating margins.
        operatingMarginsMax (float): The maximum operating margins.
        profitMarginsMin (float): The minimum profit margins.
        profitMarginsMax (float): The maximum profit margins.
        sharesOutstandingMin (float): The minimum shares outstanding.
        sharesOutstandingMax (float): The maximum shares outstanding.
        sharesFloatMin (float): The minimum shares float.
        sharesFloatMax (float): The maximum shares float.
        sharesShortMin (float): The minimum shares short.
        sharesShortMax (float): The maximum shares short.
        shortRatioMin (float): The minimum short ratio.
        shortRatioMax (float): The maximum short ratio.
        shortPercentOfFloatMin (float): The minimum short percent of float.
        shortPercentOfFloatMax (float): The maximum short percent of float.
        shortPercentOfSharesOutstandingMin (float): The minimum short percent of shares outstanding.
        shortPercentOfSharesOutstandingMax (float): The maximum short percent of shares outstanding.
        recommendationKey (str): The recommendation key.
        recommendationMeanMin (float): The minimum recommendation mean.
        recommendationMeanMax (float): The maximum recommendation mean.
    """
    filtered = stock_screener(**locals())
    return f'The following {len(filtered)} stocks meet the criteria: {", ".join(filtered)}' if filtered else 'No stocks meet the criteria'


def stock_screener(
        universe: str, currentPriceMin: str = None, currentPriceMax: str = None,
        industry: str = None, sector: str = None, country: str = None, fullTimeEmployeesMin: str = None,
        fullTimeEmployeesMax: str = None, marketCapMin: str = None, marketCapMax: str = None, betaMin: str = None,
        betaMax: str = None, dividendYieldMin: str = None, dividendYieldMax: str = None, priceToEarningsMin: str = None,
        priceToEarningsMax: str = None, forwardPEMin: str = None, forwardPEMax: str = None, priceToSalesMin: str = None,
        priceToSalesMax: str = None, priceToBookMin: str = None, priceToBookMax: str = None, revenueMin: str = None,
        revenueMax: str = None, grossProfitsMin: str = None, grossProfitsMax: str = None, freeCashFlowMin: str = None,
        freeCashFlowMax: str = None, ebitdaMin: str = None, ebitdaMax: str = None, earningsGrowthMin: str = None,
        earningsGrowthMax: str = None, revenueGrowthMin: str = None, revenueGrowthMax: str = None, debtToEquityMin: str = None,
        debtToEquityMax: str = None, currentRatioMin: str = None, currentRatioMax: str = None, quickRatioMin: str = None,
        quickRatioMax: str = None, totalAssetsMin: str = None, totalAssetsMax: str = None, totalLiabilitiesMin: str = None,
        totalLiabilitiesMax: str = None, operatingCashFlowMin: str = None, operatingCashFlowMax: str = None,
        grossMarginsMin: str = None, grossMarginsMax: str = None, operatingMarginsMin: str = None,
        operatingMarginsMax: str = None, profitMarginsMin: str = None, profitMarginsMax: str = None,
        sharesOutstandingMin: str = None, sharesOutstandingMax: str = None, sharesFloatMin: str = None,
        sharesFloatMax: str = None, sharesShortMin: str = None, sharesShortMax: str = None, shortRatioMin: str = None,
        shortRatioMax: str = None, shortPercentOfFloatMin: str = None, shortPercentOfFloatMax: str = None,
        shortPercentOfSharesOutstandingMin: str = None, shortPercentOfSharesOutstandingMax: str = None,
        recommendationKey: str = None, recommendationMeanMin: str = None, recommendationMeanMax: str = None) -> list[str]:

    if universe in [None, 'all', '']:
        universe = st.session_state.symbol_universe.index.tolist()
    else:
        universe = universe.split(',')
    universe = st.session_state.symbol_universe.loc[universe]

    # Helper function to convert string to numeric, returning None if conversion fails
    def to_numeric(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Apply filters
    df_filtered = universe

    if currentPriceMin is not None:
        df_filtered = df_filtered[df_filtered['currentPrice'] >= to_numeric(currentPriceMin)]
    if currentPriceMax is not None:
        df_filtered = df_filtered[df_filtered['currentPrice'] <= to_numeric(currentPriceMax)]

    if industry is not None:
        df_filtered = df_filtered[df_filtered['industry'] == industry]
    if sector is not None:
        df_filtered = df_filtered[df_filtered['sector'] == sector]
    if country is not None:
        df_filtered = df_filtered[df_filtered['country'] == country]

    if fullTimeEmployeesMin is not None:
        df_filtered = df_filtered[df_filtered['fullTimeEmployees'] >= to_numeric(fullTimeEmployeesMin)]
    if fullTimeEmployeesMax is not None:
        df_filtered = df_filtered[df_filtered['fullTimeEmployees'] <= to_numeric(fullTimeEmployeesMax)]

    if marketCapMin is not None:
        df_filtered = df_filtered[df_filtered['marketCap'] >= to_numeric(marketCapMin)]
    if marketCapMax is not None:
        df_filtered = df_filtered[df_filtered['marketCap'] <= to_numeric(marketCapMax)]

    if betaMin is not None:
        df_filtered = df_filtered[df_filtered['beta'] >= to_numeric(betaMin)]
    if betaMax is not None:
        df_filtered = df_filtered[df_filtered['beta'] <= to_numeric(betaMax)]

    if dividendYieldMin is not None:
        df_filtered = df_filtered[df_filtered['dividendYield'] >= to_numeric(dividendYieldMin)]
    if dividendYieldMax is not None:
        df_filtered = df_filtered[df_filtered['dividendYield'] <= to_numeric(dividendYieldMax)]

    if priceToEarningsMin is not None:
        df_filtered = df_filtered[df_filtered['priceToEarnings'] >= to_numeric(priceToEarningsMin)]
    if priceToEarningsMax is not None:
        df_filtered = df_filtered[df_filtered['priceToEarnings'] <= to_numeric(priceToEarningsMax)]

    if forwardPEMin is not None:
        df_filtered = df_filtered[df_filtered['forwardPE'] >= to_numeric(forwardPEMin)]
    if forwardPEMax is not None:
        df_filtered = df_filtered[df_filtered['forwardPE'] <= to_numeric(forwardPEMax)]

    if priceToSalesMin is not None:
        df_filtered = df_filtered[df_filtered['priceToSales'] >= to_numeric(priceToSalesMin)]
    if priceToSalesMax is not None:
        df_filtered = df_filtered[df_filtered['priceToSales'] <= to_numeric(priceToSalesMax)]

    if priceToBookMin is not None:
        df_filtered = df_filtered[df_filtered['priceToBook'] >= to_numeric(priceToBookMin)]
    if priceToBookMax is not None:
        df_filtered = df_filtered[df_filtered['priceToBook'] <= to_numeric(priceToBookMax)]

    if revenueMin is not None:
        df_filtered = df_filtered[df_filtered['revenue'] >= to_numeric(revenueMin)]
    if revenueMax is not None:
        df_filtered = df_filtered[df_filtered['revenue'] <= to_numeric(revenueMax)]

    if grossProfitsMin is not None:
        df_filtered = df_filtered[df_filtered['grossProfits'] >= to_numeric(grossProfitsMin)]
    if grossProfitsMax is not None:
        df_filtered = df_filtered[df_filtered['grossProfits'] <= to_numeric(grossProfitsMax)]

    if freeCashFlowMin is not None:
        df_filtered = df_filtered[df_filtered['freeCashFlow'] >= to_numeric(freeCashFlowMin)]
    if freeCashFlowMax is not None:
        df_filtered = df_filtered[df_filtered['freeCashFlow'] <= to_numeric(freeCashFlowMax)]

    if ebitdaMin is not None:
        df_filtered = df_filtered[df_filtered['ebitda'] >= to_numeric(ebitdaMin)]
    if ebitdaMax is not None:
        df_filtered = df_filtered[df_filtered['ebitda'] <= to_numeric(ebitdaMax)]

    if earningsGrowthMin is not None:
        df_filtered = df_filtered[df_filtered['earningsGrowth'] >= to_numeric(earningsGrowthMin)]
    if earningsGrowthMax is not None:
        df_filtered = df_filtered[df_filtered['earningsGrowth'] <= to_numeric(earningsGrowthMax)]

    if revenueGrowthMin is not None:
        df_filtered = df_filtered[df_filtered['revenueGrowth'] >= to_numeric(revenueGrowthMin)]
    if revenueGrowthMax is not None:
        df_filtered = df_filtered[df_filtered['revenueGrowth'] <= to_numeric(revenueGrowthMax)]

    if debtToEquityMin is not None:
        df_filtered = df_filtered[df_filtered['debtToEquity'] >= to_numeric(debtToEquityMin)]
    if debtToEquityMax is not None:
        df_filtered = df_filtered[df_filtered['debtToEquity'] <= to_numeric(debtToEquityMax)]

    if currentRatioMin is not None:
        df_filtered = df_filtered[df_filtered['currentRatio'] >= to_numeric(currentRatioMin)]
    if currentRatioMax is not None:
        df_filtered = df_filtered[df_filtered['currentRatio'] <= to_numeric(currentRatioMax)]

    if quickRatioMin is not None:
        df_filtered = df_filtered[df_filtered['quickRatio'] >= to_numeric(quickRatioMin)]
    if quickRatioMax is not None:
        df_filtered = df_filtered[df_filtered['quickRatio'] <= to_numeric(quickRatioMax)]

    if totalAssetsMin is not None:
        df_filtered = df_filtered[df_filtered['totalAssets'] >= to_numeric(totalAssetsMin)]
    if totalAssetsMax is not None:
        df_filtered = df_filtered[df_filtered['totalAssets'] <= to_numeric(totalAssetsMax)]

    if totalLiabilitiesMin is not None:
        df_filtered = df_filtered[df_filtered['totalLiabilities'] >= to_numeric(totalLiabilitiesMin)]
    if totalLiabilitiesMax is not None:
        df_filtered = df_filtered[df_filtered['totalLiabilities'] <= to_numeric(totalLiabilitiesMax)]

    if operatingCashFlowMin is not None:
        df_filtered = df_filtered[df_filtered['operatingCashFlow'] >= to_numeric(operatingCashFlowMin)]
    if operatingCashFlowMax is not None:
        df_filtered = df_filtered[df_filtered['operatingCashFlow'] <= to_numeric(operatingCashFlowMax)]

    if grossMarginsMin is not None:
        df_filtered = df_filtered[df_filtered['grossMargins'] >= to_numeric(grossMarginsMin)]
    if grossMarginsMax is not None:
        df_filtered = df_filtered[df_filtered['grossMargins'] <= to_numeric(grossMarginsMax)]

    if operatingMarginsMin is not None:
        df_filtered = df_filtered[df_filtered['operatingMargins'] >= to_numeric(operatingMarginsMin)]
    if operatingMarginsMax is not None:
        df_filtered = df_filtered[df_filtered['operatingMargins'] <= to_numeric(operatingMarginsMax)]

    if profitMarginsMin is not None:
        df_filtered = df_filtered[df_filtered['profitMargins'] >= to_numeric(profitMarginsMin)]
    if profitMarginsMax is not None:
        df_filtered = df_filtered[df_filtered['profitMargins'] <= to_numeric(profitMarginsMax)]

    if sharesOutstandingMin is not None:
        df_filtered = df_filtered[df_filtered['sharesOutstanding'] >= to_numeric(sharesOutstandingMin)]
    if sharesOutstandingMax is not None:
        df_filtered = df_filtered[df_filtered['sharesOutstanding'] <= to_numeric(sharesOutstandingMax)]

    if sharesFloatMin is not None:
        df_filtered = df_filtered[df_filtered['sharesFloat'] >= to_numeric(sharesFloatMin)]
    if sharesFloatMax is not None:
        df_filtered = df_filtered[df_filtered['sharesFloat'] <= to_numeric(sharesFloatMax)]

    if sharesShortMin is not None:
        df_filtered = df_filtered[df_filtered['sharesShort'] >= to_numeric(sharesShortMin)]
    if sharesShortMax is not None:
        df_filtered = df_filtered[df_filtered['sharesShort'] <= to_numeric(sharesShortMax)]

    if shortRatioMin is not None:
        df_filtered = df_filtered[df_filtered['shortRatio'] >= to_numeric(shortRatioMin)]
    if shortRatioMax is not None:
        df_filtered = df_filtered[df_filtered['shortRatio'] <= to_numeric(shortRatioMax)]

    if shortPercentOfFloatMin is not None:
        df_filtered = df_filtered[df_filtered['shortPercentOfFloat'] >= to_numeric(shortPercentOfFloatMin)]
    if shortPercentOfFloatMax is not None:
        df_filtered = df_filtered[df_filtered['shortPercentOfFloat'] <= to_numeric(shortPercentOfFloatMax)]

    if shortPercentOfSharesOutstandingMin is not None:
        df_filtered = df_filtered[df_filtered['shortPercentOfSharesOutstanding'] >=
                                  to_numeric(shortPercentOfSharesOutstandingMin)]
    if shortPercentOfSharesOutstandingMax is not None:
        df_filtered = df_filtered[df_filtered['shortPercentOfSharesOutstanding'] <=
                                  to_numeric(shortPercentOfSharesOutstandingMax)]

    if recommendationKey is not None:
        df_filtered = df_filtered[df_filtered['recommendationKey'] == recommendationKey]

    if recommendationMeanMin is not None:
        df_filtered = df_filtered[df_filtered['recommendationMean'] >= to_numeric(recommendationMeanMin)]
    if recommendationMeanMax is not None:
        df_filtered = df_filtered[df_filtered['recommendationMean'] <= to_numeric(recommendationMeanMax)]

    # Return the list of stock symbols that meet the criteria
    return df_filtered.index.tolist()


def plot_price_over_time(price_series):

    fig = plt.figure(figsize=(10, 6))
    for s in price_series:
        plt.plot(s.index, s, label=s.name)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Historical Stock Prices')
    plt.legend()
    st.pyplot(fig)


def setup_sidebar(universe=None):
    multiline_text = """
    This prototype uses the Llama 3 model to create stock index, assuming a randomly generated stock universe.
    """
    with st.sidebar:
        st.title("Index Design with Llama 3")
        st.markdown(multiline_text, unsafe_allow_html=True)
        with st.expander('Examples'):
            st.write('I want the top 10 stock with highest revenue, then top 3 with highest price to earnings ratio.')
        with st.expander("Stock Universe"):
            if universe is not None:
                c1, c2, c3 = st.columns([1, 1, 1])
                c1.metric("Stocks", universe.shape[0])
                c2.metric("Industries", universe['industry'].nunique())
                c3.metric("Countries", universe['country'].nunique())
                st.write(universe)

    log_area = st.sidebar.container()

    def log_message(message, label):
        with log_area:
            with st.expander(label):
                st.write(message)
    return log_message


def create_symbol_universe(n=100):
    symbols = generate_dummy_stock_symbols(n)
    stock_info = [get_dummy_stock_info(symbol) for symbol in symbols]
    df = pd.DataFrame(stock_info).set_index('symbol')
    return df


tool_registry = {
    'filter_stocks': filter_stocks,
    'rank_stocks_by_metric': rank_stocks_by_metric,
}


def tool_registry_dispatch(tool_name, args):
    return tool_registry[tool_name].invoke(args)


def setup_llm():
    proxy_params = {'http_client': httpx.Client(
        proxies={"http://": proxy_url, "https://": proxy_url})} if proxy_url else {}
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model='llama3-70b-8192',
        temperature=0.0,
        **proxy_params
    )
    return llm.bind_tools(tool_registry.values(), tool_choice='auto')


greeting_prompt = '''
Designing a stock index involves answering questions like: 
1. **Purpose**: What is the primary objective of this index? 
2. **Geographical Focus**: Which region or regions will the index cover?
3. **Market Focus**: What type of equities will be included, i.e. size, sector, industry?
4. **Inclusion Criteria**: What criteria will you use to select the companies included in the index, such as trading volume, capticalization, financial metrics?
5. **Exclusion Criteria**: Are there any companies or sectors you want to exclude?
6. **Weighting Method**: How will the companies be weighted in the index?
7. **Rebalancing Frequency**: How often will the index be reviewed and rebalanced?
8. **Calculation Type**: How will dividends, coupons, taxes be treated in the index calculation?
9. **Base Date and Base Value**: What will be the base date and the base value for the index?

Ask me anything about stock indices!
'''

template = '''
You are helping the user design a stock index. The stock universe is given. 
Answer the questions as you can and use the tools to interact with the stock universe.
Use the tools only when necessary. Use the arguments that the user provides.

What a tool returns unknown or zero result, just say "I don't know" or "I can't find it".

Use the following format:

Thought: you should always think about what to do

Action: the action you take

Observation: the result of the tool call

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

'''


def main():

    if st.session_state.get('symbol_universe') is None:
        st.session_state.symbol_universe = create_symbol_universe()

    logger = setup_sidebar(st.session_state.symbol_universe)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": greeting_prompt},
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    llm = setup_llm()

    if question := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        messages = [
            SystemMessage(template),
            HumanMessage(question)
        ]
        with st.status("Thinking...", expanded=True):
            while True:
                ai_msg = llm.invoke(messages)
                messages.append(ai_msg)
                if ai_msg.response_metadata['finish_reason'] == 'tool_calls':
                    logger(ai_msg.dict(), 'Tool Call')
                    for tool_call in ai_msg.tool_calls:
                        call_prompt = f'Calling tool: {tool_call["name"]}({tool_call["args"]})'
                        st.code(call_prompt)
                        tool_output = tool_registry_dispatch(tool_call["name"], tool_call["args"])
                        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
                else:
                    break

        logger(ai_msg.dict(), 'Final')

        with st.chat_message("assistant"):
            st.markdown(ai_msg.content)
        st.session_state.messages.append({"role": "assistant", "content": ai_msg.content})


if __name__ == "__main__":
    main()
