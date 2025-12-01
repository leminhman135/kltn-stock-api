# src/model.py
from statsmodels.tsa.arima.model import ARIMA

def build_arima_model(df):
    model = ARIMA(df['price'], order=(5,1,0))
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=10):
    forecast = model_fit.forecast(steps=steps)
    return forecast
