import copy
import functools
import os
import random
import string
from datetime import date
from typing import List, Optional

import httpx
import jsonschema
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from groq import BadRequestError
from langchain.globals import set_debug
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from scipy.stats import norm

set_debug(True)

proxy_url = os.getenv('GROQ_PROXY') or None

json_schema = {
    "type": "object",
    "properties": {
        "filters": {
            "type": "object",
            "properties": {
                "market_cap": {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "array",
                            "items": {"type": "number", "maximum": 1e100},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "percentile_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Array containing min and max percentiles for filtering in range 0 - 100"
                        }
                    }
                },
                "pe_ratio": {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "percentile_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Array containing min and max percentiles for filtering in range 0 - 100"
                        }
                    }
                },
                "dividend_yield": {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "percentile_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Array containing min and max percentiles for filtering in range 0 - 100"
                        }
                    }
                },
                "debt_equity_ratio": {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 1,
                            "maxItems": 2
                        },
                        "percentile_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 1,
                            "maxItems": 2,
                            "description": "Array containing min and max percentiles for filtering in range 0 - 100"
                        }
                    }
                },
                "esg_score": {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "percentile_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Array containing min and max percentiles for filtering in range 0 - 100"
                        }
                    }
                },
                "price_to_book_ratio": {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "percentile_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Array containing min and max percentiles for filtering in range 0 - 100"
                        }
                    }
                },
                "revenue": {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "percentile_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Array containing min and max percentiles for filtering in range 0 - 100"
                        }
                    }
                },
                "earnings_per_share": {
                    "type": "object",
                    "properties": {
                        "range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "percentile_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "Array containing min and max percentiles for filtering in range 0 - 100"
                        }
                    }
                },
                "sector": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["Technology", "Healthcare", "Finance", "Consumer Discretionary",
                                                         "Energy", "Utilities", "Materials", "Industrials", "Real Estate",
                                                         "Communication Services"]}
                },
                "country": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["USA", "Canada", "UK", "Germany", "France", "Japan", "China",
                                                        "India", "Brazil", "Australia"]}
                }
            }
        },
        "sort_by": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "enum": ["market_cap", "pe_ratio", "dividend_yield", "debt_equity_ratio",
                                                     "esg_score", "price_to_book_ratio", "revenue", "earnings_per_share"]},
                "order": {"type": "string", "enum": ["asc", "desc"]}
            },
            "required": ["field", "order"]
        },
        "limit": {
            "type": "integer",
            "minimum": 1
        },
        "percentile_range_limit": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
            "description": "Array containing min and max percentiles to select from the results"
        },
        "weighting_method": {
            "type": "string",
            "enum": ["market_cap", "equal"],
            "description": "Specifies how companies are weighted in the index"
        },
        "rebalancing_frequency": {
            "type": "string",
            "enum": ["weekly", "monthly", "quarterly"],
            "description": "How often the index is reviewed and rebalanced"
        },
        "calculation_type": {
            "type": "string",
            "enum": ["total_return", "price_return", "net_return"],
            "description": "How dividends, coupons, and taxes are treated in the index calculation"
        },
        "base_date": {
            "type": "string",
            "format": "date",
            "description": "The date at which the index base value is set"
        },
        "base_value": {
            "type": "number",
            "description": "The starting value of the index at the base_date"
        }
    },
    "required": []
}


def verify_schema(json):
    try:
        jsonschema.validate(instance=json, schema=json_schema)
    except Exception:
        st.code("Invalid config detected. Attempting to fix it.")
        if json.get('sector') and json['sector'].get('enum'):
            json['sector'] = json['sector']['enum']
        if json.get('country') and json['country'].get('enum'):
            json['country'] = json['country']['enum']

def generate_stock_metadata(num_stocks):
    def random_ticker():
        return ''.join(random.choices(string.ascii_uppercase, k=3))

    tickers = [random_ticker() for _ in range(num_stocks)]
    market_caps = np.random.uniform(1e9, 2e12, num_stocks)  # Market cap between 1 billion and 2 trillion USD
    pe_ratios = np.random.uniform(5, 50, num_stocks)  # PE ratio between 5 and 50
    dividend_yields = np.random.uniform(0, 0.1, num_stocks)  # Dividend yield between 0% and 10%
    debt_equity_ratios = np.random.uniform(0, 2, num_stocks)  # Debt-to-equity ratio between 0 and 2
    esg_scores = np.random.uniform(50, 90, num_stocks)  # ESG score between 50 and 90
    price_to_book_ratios = np.random.uniform(1, 15, num_stocks)  # Price-to-book ratio between 1 and 15
    revenues = np.random.uniform(1e8, 1e11, num_stocks)  # Revenue between 100 million and 100 billion USD
    earnings_per_share = np.random.uniform(0.5, 10, num_stocks)  # Earnings per share between 0.5 and 10
    sectors = [
        random.choice(
            ["Technology", "Healthcare", "Finance", "Consumer Discretionary", "Energy", "Utilities", "Materials",
             "Industrials", "Real Estate", "Communication Services"]) for _ in range(num_stocks)]
    countries = [
        random.choice(["USA", "Canada", "UK", "Germany", "France", "Japan", "China", "India", "Brazil", "Australia"])
        for _ in range(num_stocks)]

    data = {
        "market_cap": market_caps,
        "pe_ratio": pe_ratios,
        "dividend_yield": dividend_yields,
        "debt_equity_ratio": debt_equity_ratios,
        "esg_score": esg_scores,
        "price_to_book_ratio": price_to_book_ratios,
        "revenue": revenues,
        "earnings_per_share": earnings_per_share,
        "sector": sectors,
        "country": countries
    }

    df = pd.DataFrame(data, index=tickers)
    return df


def random_update_stock_metadata(df):
    """
    Apply random updates to stock metadata in a DataFrame to simulate realistic changes.

    Parameters:
    - df: DataFrame containing stock metadata.

    Returns:
    - DataFrame with updated metadata.
    """
    # Define realistic ranges for each field to ensure updates are plausible
    ranges = {
        'market_cap': (1e9, 1e12),  # Market Cap in range from 1 Billion to 1 Trillion
        'pe_ratio': (5, 50),        # P/E Ratio in range from 5 to 50
        'dividend_yield': (0, 10),  # Dividend Yield in range from 0% to 10%
        'debt_equity_ratio': (0, 2),  # Debt/Equity Ratio from 0 to 2
        'esg_score': (0, 100),       # ESG Score from 0 to 100
        'price_to_book_ratio': (0, 5),  # Price to Book Ratio from 0 to 5
        'revenue': (1e8, 1e11),      # Revenue from 100 Million to 100 Billion
        'earnings_per_share': (-10, 50)  # EPS from -10 to 50
    }

    # Function to apply random updates within the defined ranges
    def apply_random_update(x, field):
        if field in ranges:
            min_val, max_val = ranges[field]
            # Apply a random change within a reasonable percentage of the current value
            change_percent = np.random.uniform(-0.05, 0.05)  # Random change between -5% and +5%
            new_value = x * (1 + change_percent)
            return np.clip(new_value, min_val, max_val)
        else:
            return x  # If field is not defined in ranges, no change is applied

    # Update each field in the DataFrame using the random update function
    for field in ranges.keys():
        if field in df.columns:
            df[field] = df[field].apply(lambda x: apply_random_update(x, field))

    return df


def generate_random_walk_prices(symbols):
    """
    Generate dummy stock close prices following a random walk for a list of stock symbols up to today.

    Parameters:
    - symbols: List of stock symbols to generate prices for.

    Returns:
    - DataFrame with stock symbols as columns and rows as dates with generated close prices.
    """
    end_date = pd.Timestamp.today()
    start_date = pd.Timestamp('2023-01-01')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize DataFrame to hold the stock prices
    price_df = pd.DataFrame(index=date_range)

    # Generate random walk prices for each symbol
    for symbol in symbols:
        # Randomly initialize the start price between 50 and 150
        start_price = np.random.uniform(50, 150)

        # Initialize the price series with the start price
        prices = [start_price]

        # Generate random daily returns and accumulate them
        for _ in range(1, len(date_range)):
            daily_return = np.random.normal(loc=0.001, scale=0.02)  # Daily return with mean 0.1% and std dev 2%
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)

        # Add the price series to the DataFrame
        price_df[symbol] = prices

    return price_df


def update_schema_object(config=None, update=None, ):
    def recursive_merge(dict1, dict2):
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                recursive_merge(dict1[key], value)
            else:
                # Merge non-dictionary values
                dict1[key] = value
    if not update:
        return config
    else:
        config = copy.deepcopy(config) if config is not None else {}
        recursive_merge(config, update)
        return config


def apply_screening(metadata, config):
    # Extract filters from the schema
    filters = config.get('filters', {})
    progress = []

    # Apply percentile range filters first
    for field, conditions in filters.items():
        if 'percentile_range' in conditions and pd.api.types.is_numeric_dtype(metadata[field]):
            min_percentile, max_percentile = conditions['percentile_range']
            min_value = metadata[field].quantile(min_percentile / 100.0)
            max_value = metadata[field].quantile(max_percentile / 100.0)
            metadata = metadata[(metadata[field] >= min_value) & (metadata[field] <= max_value)]
            progress.append((field, 'percentile_range', len(metadata)))

    # Apply numeric range filters
    for field, conditions in filters.items():
        if 'range' in conditions and pd.api.types.is_numeric_dtype(metadata[field]):
            min_value, max_value = conditions['range']
            if min_value is not None:
                metadata = metadata[metadata[field] >= min_value]
            if max_value is not None:
                metadata = metadata[metadata[field] <= max_value]
            progress.append((field, 'range', len(metadata)))

    # Apply categorical filters
    for field in ['sector', 'country']:
        if field in filters:
            allowed_values = filters[field]
            if allowed_values:
                metadata = metadata[metadata[field].isin(allowed_values)]
            progress.append((field, allowed_values, len(metadata)))

    # Apply sorting
    sort_by = config.get('sort_by', {})
    field = sort_by.get('field')
    order = sort_by.get('order', 'asc')
    if field in metadata.columns:
        metadata = metadata.sort_values(by=field, ascending=(order == 'asc'))

    # Apply limit
    limit = config.get('limit', None)
    if limit is not None:
        metadata = metadata.head(limit)
        progress.append(('limit', limit, len(metadata)))

    # Apply percentile range limit (if provided)
    percentile_range_limit = config.get('percentile_range_limit', None)
    if percentile_range_limit:
        min_percentile, max_percentile = percentile_range_limit
        for field in metadata.columns:
            if pd.api.types.is_numeric_dtype(metadata[field]):
                min_value = metadata[field].quantile(min_percentile / 100.0)
                max_value = metadata[field].quantile(max_percentile / 100.0)
                metadata = metadata[(metadata[field] >= min_value) & (metadata[field] <= max_value)]
                progress.append(('percentile_range_limit', field, len(metadata)))

    print(progress)
    return metadata.index.to_list()


def calculate_index_value(stock_price_df, stock_metadata, schema):
    """
    Calculate the index value based on index symbols, stock price DataFrame, stock metadata, and schema.

    Parameters:
    - index_symbols: List of stock symbols to be included in the index.
    - stock_price_df: DataFrame with historical stock prices.
    - stock_metadata: DataFrame with stock metadata including tickers, market cap, etc.
    - schema: Dictionary with index definition details including weighting method and rebalancing frequency.

    Returns:
    - DataFrame with index values over time.
    """
    # Extract schema details
    weighting_method = schema.get('weighting_method', 'market_cap')
    rebalancing_frequency = schema.get('rebalancing_frequency', 'monthly')

    # Filter metadata for the index symbols
    stock_metadata = random_update_stock_metadata(stock_metadata)
    index_symbols = apply_screening(stock_metadata, schema)
    relevant_metadata = stock_metadata[stock_metadata.index.isin(index_symbols)].copy()

    # Define rebalancing period
    rebalance_period = {
        'quarterly': 'Q',
        'monthly': 'M',
        'weekly': 'W'
    }.get(rebalancing_frequency, 'M')

    def calculate_weights(metadata, method):
        """Calculate weights based on the specified weighting method."""
        if method == 'market_cap':
            metadata['weight'] = metadata['market_cap'] / metadata['market_cap'].sum()
        elif method == 'equal':
            metadata['weight'] = 1.0 / len(metadata)
        else:
            raise ValueError(f"Unsupported weighting method: {method}")
        return metadata[['weight']]

    def apply_weights(prices, weights):
        """Apply weights to stock prices to calculate the index value."""
        weighted_prices = prices * weights
        return weighted_prices.sum()

    def rebalance_weights(date):
        """Calculate weights for the rebalance date."""
        # Recalculate weights on rebalancing dates
        return calculate_weights(relevant_metadata, weighting_method)

    # Initialize DataFrame to hold the index values
    index_values = pd.DataFrame(index=stock_price_df.index)

    # Initialize weights
    current_weights = calculate_weights(relevant_metadata, weighting_method)

    # Calculate index values
    last_rebalance_date = stock_price_df.index[0]

    for date in stock_price_df.index:
        # Rebalance if needed
        dr = pd.date_range(last_rebalance_date, date, freq=rebalance_period)
        if len(dr) > 0 and dr[-1] == date:
            stock_metadata = random_update_stock_metadata(stock_metadata)
            index_symbols = apply_screening(stock_metadata, schema)
            relevant_metadata = stock_metadata[stock_metadata.index.isin(index_symbols)].copy()
            current_weights = rebalance_weights(date)
            last_rebalance_date = date

        # Calculate index value
        if not current_weights.empty:
            index_values.loc[date, 'Index'] = apply_weights(stock_price_df.loc[date], current_weights['weight'])
        else:
            index_values.loc[date, 'Index'] = np.nan  # Handle case where no weights are available

    return index_values


def calculate_all_periods_performance(index_values, base_value=1000):
    # Ensure the index_values is a DataFrame with a DateTime index
    if not isinstance(index_values, pd.DataFrame):
        raise ValueError("index_values should be a pandas DataFrame")
    if not isinstance(index_values.index, pd.DatetimeIndex):
        raise ValueError("Index of index_values should be a pandas DateTimeIndex")

    periods = {
        '30D': pd.DateOffset(days=30),
        '90D': pd.DateOffset(days=90),
        '180D': pd.DateOffset(days=180),
        '360D': pd.DateOffset(days=360),
        'YTD': pd.DateOffset(months=(pd.Timestamp.now().month - 1)),
        'Since Inception': pd.DateOffset(days=(pd.Timestamp.now() - index_values.index.min()).days)
    }

    def calculate_period_metrics(df, period):
        start_date = df.index[-1] - period
        df_period = df.loc[start_date:]

        if df_period.empty:
            return {metric: np.nan for metric in metrics_names}

        # Calculate daily returns
        df_period['Return'] = df_period.pct_change().dropna()

        # Metrics
        metrics = {}
        metrics['Performance'] = (df_period.iloc[-1] / df_period.iloc[0] - 1).values[0]
        metrics['Volatility (p.a.)'] = df_period['Return'].std() * np.sqrt(252)  # Annualize
        metrics['High'] = df_period.max().values[0]
        metrics['Low'] = df_period.min().values[0]
        metrics['Sharpe Ratio'] = metrics['Performance'] / metrics['Volatility (p.a.)']
        metrics['Max Drawdown'] = ((df_period / df_period.cummax() - 1).min()).values[0]

        # VaR and CVaR
        returns = df_period['Return'].dropna()
        metrics['VaR 95'] = np.percentile(returns, 5)
        metrics['CVaR 95'] = returns[returns <= metrics['VaR 95']].mean()

        return metrics

    metrics_names = ['Performance', 'Volatility (p.a.)', 'High', 'Low',
                     'Sharpe Ratio', 'Max Drawdown', 'VaR 95', 'CVaR 95']
    results = []

    for period_name, period in periods.items():
        metrics = calculate_period_metrics(index_values, period)
        metrics['Period'] = period_name
        results.append(metrics)

    return pd.DataFrame(results).set_index('Period').T


@tool
def overwrite_config(update):
    ''' Overwrite the current configuration with a new one. This should be called when existing config should be scratched.

    Do this when the user wants to start over, reset, or when the user wants to change the configuration completely

    update: The new configuration to set. It must follow the config schema. sector and country are list.
    '''
    verify_schema(update)
    st.session_state.old_config = copy.deepcopy(st.session_state.config)
    st.session_state.config = copy.deepcopy(update)
    # st.write(st.session_state.old_config, st.session_state.config)
    result = apply_screening(st.session_state.symbol_metadata, st.session_state.config)
    return f'There are {len(result)} stocks selected so far.' if result else 'No result.'
    

@tool
def update_config(update):
    ''' Filter stocks based on the current configuration. This should be called when the configuration is updated.

    update: The updated configuration to apply to the current configuration. It must follow the config schema. sector and country are list.
    '''
    verify_schema(update)
    st.session_state.old_config = copy.deepcopy(st.session_state.config)
    st.session_state.config = update_schema_object(st.session_state.config, update)
    # st.write(st.session_state.old_config, st.session_state.config)
    result = apply_screening(st.session_state.symbol_metadata, st.session_state.config)
    st.code(f'{len(result)} stocks meet the criteria: {", ".join(result)}')
    return f'There are {len(result)} stocks selected.' if result else 'No result.'


def setup_sidebar(universe=None):
    multiline_text = """
    This prototype uses the Llama 3 model to create stock index, assuming a randomly generated stock universe.
    """
    with st.sidebar:
        st.title("Index Design with Llama 3")
        st.markdown(multiline_text, unsafe_allow_html=True)
        with st.expander('Examples'):
            st.caption('An index tracking the performance of renewable energy companies in Europe, with a focus on market leaders and innovators.')
            st.caption('An index focusing on high-growth companies in developing countries, targeting sectors like technology, healthcare, and consumer goods.')
        with st.expander("Stock Universe"):
            if universe is not None:
                c1, c2, c3 = st.columns([1, 1, 1])
                c1.metric("Stocks", universe.shape[0])
                c2.metric("Sectors", universe['sector'].nunique())
                c3.metric("Countries", universe['country'].nunique())
                st.write(universe)
        with st.expander("Configuration"):
            st.button("Refresh Configuration")
            st.write(st.session_state.config)

    log_area = st.sidebar.container()

    def log_message(message, label):
        with log_area:
            with st.expander(label):
                st.write(message)
    return log_message


def tool_registry_dispatch(tool_name, args):
    return tool_registry[tool_name].invoke(args)


def setup_llm():
    proxy_params = {'http_client': httpx.Client(
        proxies={"http://": proxy_url, "https://": proxy_url})} if proxy_url else {}
    llm = ChatGroq(
        groq_api_key=os.getenv('GROQ_API_KEY'),
        model='llama3-70b-8192',
        temperature=0.,
        **proxy_params
    )
    return llm.bind_tools(tool_registry.values(), tool_choice='auto')


def backtest():
    index_series = calculate_index_value(st.session_state.symbol_price,
                                         st.session_state.symbol_metadata, st.session_state.config)
    st.line_chart(index_series)
    index_performance = calculate_all_periods_performance(index_series)
    st.write(index_performance)


@tool
def run_backtest():
    ''' Run backtest based on the current configuration. This should be called when the configuration is finalized.'''
    backtest()
    return 'Backtest completed.'


tool_registry = {
    'update_config': update_config,
    'overwrite_config': overwrite_config,
    'run_backtest': run_backtest,
}

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

I will guide you to design the index by asking questions. When you are ready, say "I am done" and we will run a backtest to evaluate the index performance.

Let's get started!
'''

template = '''
You are helping the user design a stock index. The index configuration must follow this schema: {schema}

The current config is: {config}

Guide user to design the index by asking questions and manipulating the schema. If user uses metric not supported by the schema, say the metric is not supported.

If user wants to start over, reset, or change the configuration completely, call the tool "overwrite_config" with the new configuration.

It's OK if zero stocks are selected.

Use tools when you feel confident. When the tool returns error or zero result, stop using the tool and continue with the next question.

When the user signals he has finished, run backtest with current config whatever it is. When the backtest is done, summarize the key features of the index in one sentence.


Output using the following format:

**Thought**: you should always think about what to do

**Action**: the action you take, omit if none

**Observation**: the result of the tool call, omit if none

... (this Thought/Action/Action Input/Observation can repeat N times)

**Answer** answer to the original input question

'''


def main():

    if 'symbol_metadata' not in st.session_state:
        st.session_state.symbol_metadata = generate_stock_metadata(1000)
        st.session_state.symbol_price = generate_random_walk_prices(st.session_state.symbol_metadata.index.tolist())
        st.session_state.config = {}
        st.sidebar.info('Welcome to Index Design')

    logger = setup_sidebar(st.session_state.symbol_metadata)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": greeting_prompt},
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    llm = setup_llm()

    try:
        if question := st.chat_input("Let's design an index"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            messages = [
                SystemMessage(template.format(schema=json_schema, config=st.session_state.config)),
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
    except BadRequestError as e:
        st.error(f"LLM returns error: {e}")


if __name__ == "__main__":
    main()
