import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 
  
# Read the AirPassengers dataset 
# df = pd.read_csv("C:\\MISC\\IU_downloads\\Thesis\\test data_time_series_crankshaft_v1.3.csv")
df = pd.read_csv("C:\\MISC\\IU_downloads\\Thesis\\test data_time_series_forged_plate_v1.3.csv")

# df = pd.read_csv("C:\\Users\\LENOVO\\Downloads\\CPI_Urban_US_City_avg.csv") 
  # 
# Print the first five rows of the dataset 
print(df) 
  
# ETS Decomposition 
result = seasonal_decompose(df['Stress'],  
                            model ='additive',period=len(df["Node_No"])//2) 
# result=seasonal_decompose(df["Stress"], model='multiplicative', period=len(df["Node_No"])//2)
  
# ETS plot  
result.plot()
plt.show()
# Import the library 
from pmdarima import auto_arima 
from statsmodels.tsa.arima.model import ARIMA
  
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 
  
# Fit auto_arima function to AirPassengers dataset

stepwise_fit = auto_arima(df["Stress"], start_p = 0, start_q = 0, 
                          max_p = 2, max_q = 2, m = 2, 
                          start_P = 0, seasonal = True, 
                          d = None, D =1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 
  
# To print the summary 
stepwise_fit.summary()
plt.rcParams['figure.figsize'] = (25,15)
ARIMA_CPI = ARIMA(np.asarray(df["Stress"]), order=(0, 2, 1), trend="n").fit()
ARIMA_CPI.summary()
ARIMA_CPI.plot_diagnostics()
plt.show()