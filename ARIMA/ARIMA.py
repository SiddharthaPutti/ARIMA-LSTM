
import pandas as pd
import numpy as np
import datetime
import copy
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle

sns.set_style("whitegrid", {"font.sans-serif": ['KaiTi', 'Arial']})

'''
fr=open('StockFile.pkl','rb')
data1=pickle.load(fr)
df,num_stock,stock_code=data1[0],data1[1],data1[2]
'''
# the argument
FREQ = '3D'


# fr=open('StockFile.pkl','rb')
# data1=pickle.load(fr)
# df,stock_code=data1[0],data1[1]

f = open('traing_data_new.csv')
df = pd.read_csv(f)  
df1=df.iloc[:100000,:]
df1['datetime']=df1['Date']+'-'+df1['Time']
df1.index=df1['datetime']
df1.drop(df1.columns[[0,1,2,3,5,9]], axis=1, inplace=True)
df1.columns=['open','high','low','close']
def run_main():
    
    data=df1
    d_one = data.index  
    d_two = []
    d_three = []
    date2 = []
    for i in d_one:
        d_two.append(i)
    for i in range(len(d_two)):
        d_three.append(parse(d_two[i]))
    data2 = pd.DataFrame(data, index=d_three,
                         dtype=np.float64)  

    data2 = data2.drop_duplicates(keep='first')
    data2 = data2.sort_index(axis=0)
    plt.plot(data2['close'])
    
    plt.title('plot')
    #plt.show()

    data2_w = data2['close'].resample(FREQ).mean()
    data2_train = data2_w['2008-01':'2009-11']  
    plt.plot(data2_train)
    plt.title('plot')
    #plt.show()
    data2_train = data2_train.dropna(axis=0, how='any')
    new_index = pd.date_range('20180101', periods=len(data2_train),freq = FREQ)
    data2_train = pd.DataFrame(data2_train)
    data_train = copy.copy(data2_train)
    data2_train.set_index(new_index, inplace=True)
    acf = plot_acf(data2_train, lags=20)  
    plt.title("stock index ACF")
    # acf.show()
    pacf = plot_pacf(data2_train, lags=20)  
    plt.title("stock index PACF")
    # pacf.show()
    data2_diff = data2_train.diff(1)  
    diff = data2_diff.dropna()
    for i in range(1): 
        diff = diff.diff(1)
        diff = diff.dropna()
    plt.figure()
    plt.plot(diff)
    plt.title('2 order difference')
    #plt.show()

    # ACF
    acf_diff = plot_acf(diff, lags=40)
    plt.title("ACF")  
    # acf_diff.show()

    
    pacf_diff = plot_pacf(diff, lags=40)  
    plt.title("PACF")
    pacf_diff.show()

    
    data2_train_fit = data2_train[0:(len(data2_train) - 30)]
    model = ARIMA(data2_train_fit, order=(5, 2, 2), freq=FREQ)

    pred_begin = pd.date_range('20180101', periods=len(data2_train) - 30, freq = FREQ)[-1]   
    arima_result = model.fit()

    
    forcast_vals = arima_result.forecast(30)[0]
    fore_new_index = pd.date_range(pred_begin, periods=len(forcast_vals))
    forcast_vals = pd.DataFrame(forcast_vals)
    forcast_vals.set_index(fore_new_index, inplace=True)
    
    fore_stock_forcast = pd.concat([data2_train, forcast_vals], axis=1,
                                   keys=['original', 'predicted']) 
   
    plt.figure()
    plt.plot(fore_stock_forcast)
    plt.title("Forcast Results")
    plt.show()
    a  = 1


if __name__ == "__main__":
    run_main()