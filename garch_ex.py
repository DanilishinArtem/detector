# Import necessary libraries
import numpy as np
import pandas as pd
import arch
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

# Generate example data
np.random.seed(0)
returns = np.random.randn(50)
print("Part arima model")
# Fit ARIMA(1,1) model
arima_model = ARIMA(returns, order=(1,1,0))
arima_result = arima_model.fit()
print("Part garch model")
# Fit GARCH(1,1) model
garch_model = arch_model(returns, vol='Garch', p=1, q=1)
garch_result = garch_model.fit(disp='off')

# Forecast one step ahead
arima_forecast = arima_result.forecast(steps=1)
garch_forecast = garch_result.forecast(horizon=1)

# Calculate confidence interval
confidence_interval = garch_forecast.mean - np.sqrt(garch_forecast.variance) * 1.96, garch_forecast.mean + np.sqrt(garch_forecast.variance) * 1.96
print("[ " + str(str(confidence_interval[0].values[0][0])) + " , " + str(str(confidence_interval[1].values[0][0])) + " ]")