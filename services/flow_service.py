from langgraph.graph import StateGraph
from prophet import Prophet
from typing_extensions import TypedDict
import pandas as pd
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv('GEMINI_API_KEY')

today = date.today()
# Define the shared state structure
class FinancialAnalysisState(TypedDict):
    ticker: str
    period: str
    interval: str
    prompt: str
    stock_data: dict
    predictions: list
    report: str



def extract_data(state: dict) -> dict:
    prompt = state['prompt'] 
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY, max_tokens=50)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Extract the stock ticker, period, and interval from the following prompt. "
                   "Return your answer as a JSON object with keys 'ticker', and 'period', period should be the number of days, period should be a number only"),
        ("user", "{user_input}")
    ])
    formatted_prompt = prompt_template.format_prompt(user_input=prompt)
    response = llm.invoke(formatted_prompt.to_messages()).content

    # Remove any text before the first '{' and after the last '}'
    json_text_match = re.search(r'\{.*\}', response, re.DOTALL)

    if json_text_match:
        json_text = json_text_match.group()

        try:
            details = json.loads(json_text)
        except json.JSONDecodeError:
            details = {}
    else:
        details = {}

    state['ticker'] = details.get("ticker")
    state['period'] = details.get("period")
    return state



def process_stock_data(state: FinancialAnalysisState) -> FinancialAnalysisState:
    ticker = state['ticker']
    period = '1y'
    interval = '1d'

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    df.reset_index(inplace=True)

    if 'Date' in df.columns:
        df.rename(columns={'Date': 'ds'}, inplace=True)
    else:
        df.rename(columns={'index': 'ds'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    if 'Close' in df.columns:
        df.rename(columns={'Close': 'y'}, inplace=True)
    elif 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'y'}, inplace=True)
    else:
        raise ValueError("Dataframe does not have a 'Close' or 'Adj Close' column.")

    state['stock_data'] = df.to_dict(orient='list')
    return state


def predict_stock_movement(state: FinancialAnalysisState) -> FinancialAnalysisState:
    df = pd.DataFrame(state['stock_data'])

    if 'ds' not in df.columns or 'y' not in df.columns:
        raise ValueError("Dataframe must have columns 'ds' and 'y'.")

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Create a DataFrame for future dates (forecasting 7 days ahead)
    default_period = 7
    period_value = state['period'] if state['period'] is not None else default_period
    future = model.make_future_dataframe(periods=period_value)
    forecast = model.predict(future)

    # Extract the last 7 days of forecasts
    forecast_subset = forecast[['ds', 'yhat']].tail(period_value)
    state['predictions'] = forecast_subset.to_dict(orient='list')
    return state


def generate_report(state: FinancialAnalysisState) -> FinancialAnalysisState:
    prompt = state['prompt']
    predictions = state['predictions']
    df = pd.DataFrame(state['stock_data'])
    last_price = df['y'].iloc[-1]
    avg_price = df['y'].mean()
    min_price = df['y'].min()
    max_price = df['y'].max()

    # Convert predictions to a string representation that is more readable in the report
    predictions_str = "\n".join([f"  {pd.to_datetime(pred['ds']).date()}: {pred['yhat']:.2f}" for pred in [dict(zip(predictions.keys(), values)) for values in zip(*predictions.values())]]) # Convert predictions to string
    default_period = 7
    period_value = state['period'] if state['period'] is not None else default_period


    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
            "You are a financial analyst. Based solely on the following historical stock data and forecast predictions, "
            f"please generate a detailed report that summarizes recent market trends and provides actionable insights for investors. Future predictions start from tomorrow. Note that today's date is {today}\n\n"
            "Historical Data Summary:\n"
            f"stock: {state['ticker']}"
            f"- Last Closing Price: {last_price:.2f}\n"
            f"- Average Closing Price: {avg_price:.2f}\n"
            f"- Minimum Closing Price: {min_price:.2f}\n"
            f"- Maximum Closing Price: {max_price:.2f}\n\n"
            f"Predicted Returns for the Next {period_value} Days:\n{predictions_str}\n\n"
            "Provide your analysis on potential market movements and any recommendations for investors."
            ),
            ("user", "{user_input}"),
        ]
    )

    # Format the prompt template with user input
    formatted_prompt = prompt_template.format_prompt(user_input='give me the summary on the stock')

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key='AIzaSyDk3iIRV2N4mA7wpyHnbe1pjcjXtvfcizE', max_tokens=500)

    # Pass the formatted prompt string to invoke
    report = llm.invoke(formatted_prompt.to_messages())
    state['report'] = report.content
    return state

def plot_stock_trend(state: FinancialAnalysisState) -> FinancialAnalysisState:
    default_period = 7
    period_value = state['period'] if state['period'] is not None else default_period

    # Convert historical stock data to DataFrame
    df_historical = pd.DataFrame(state['stock_data'])
    # Convert predictions dictionary to DataFrame
    df_predictions = pd.DataFrame(state['predictions'])

    # Convert 'ds' columns to datetime format
    df_historical['ds'] = pd.to_datetime(df_historical['ds'])
    df_predictions['ds'] = pd.to_datetime(df_predictions['ds'])

    # Concatenate historical and predicted data for x-axis
    df_historical_last_7 = df_historical.tail(period_value)
    df_combined_x = pd.concat([df_historical_last_7, df_predictions], ignore_index=True)
    # Corrected: use df_combined_x instead of undefined df_combined
    df_combined_x['ds'] = pd.to_datetime(df_combined_x['ds'])

    # Select only the last 7 days of historical data for y-axis
    df_historical_last_7 = df_historical.tail(period_value)
    # Rename columns for consistency
    df_historical_last_7 = df_historical_last_7[['ds', 'y']].rename(columns={'y': 'Price'})
    df_predictions = df_predictions.rename(columns={'yhat': 'Price'})
    
    df_combined_y = pd.concat([df_historical_last_7, df_predictions], ignore_index=True)

    # Set predictions with lists for x and y, e.g., dates and prices
    state["predictions"] = {
        "x": df_combined_x['ds'].dt.strftime('%Y-%m-%d').tolist(),
        "y": df_combined_y['Price'].tolist()
    }

    print('done')
    return state



# Build the state graph
def build_graph():
    workflow = StateGraph(FinancialAnalysisState)

    workflow.add_node("data_ingestion", process_stock_data)
    workflow.add_node("predict_stock", predict_stock_movement)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("visualize_data", plot_stock_trend)
    workflow.add_node('extract_data', extract_data)

    workflow.add_edge("data_ingestion", "predict_stock")
    workflow.add_edge("predict_stock", "generate_report")
    workflow.add_edge("generate_report", "visualize_data")
    workflow.add_edge("extract_data", "data_ingestion")

    workflow.set_entry_point("extract_data")
    workflow.set_finish_point("visualize_data")

    return workflow


graph = build_graph()
app = graph.compile()

