import pmdarima as pm

def train_arima(series):
    model = pm.auto_arima(series, seasonal=True, m=7)
    return model
