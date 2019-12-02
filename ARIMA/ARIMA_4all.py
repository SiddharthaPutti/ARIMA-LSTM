
import pandas as pd
import numpy as np
import datetime
import copy
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
import scipy.interpolate as itp
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
df_results = pd.DataFrame()

# fr=open('StockFile.pkl','rb')
# data1=pickle.load(fr)
# df,stock_code=data1[0],data1[1]

a = 1
def run_main( stock_index, df,stock_code):
    #df1 = df.iloc[:100000, :]
    print(stock_code[stock_index])
    df1 = pd.DataFrame(df[stock_index][:, :],
                       columns=['index', 'stock_code', 'Date', 'Time', 'open', 'what', 'high', 'low', 'close'])
    df1['datetime'] = df1['Date'] + '-' + df1['Time']
    df1.index = df1['datetime']
    df1.drop(df1.columns[[0, 1, 2, 3, 5, 9]], axis=1, inplace=True)
    df1.columns = ['open', 'high', 'low', 'close']
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
    plt.title('dialy closing price of stock market ')
    #plt.show()

    data2_w = data2['close'].resample(FREQ).mean()
    data2_train = data2_w 
    data2_train = data2_train.dropna(axis=0, how='any')
    new_index = pd.date_range('20180101', periods=len(data2_train),freq = FREQ)
    data2_train = pd.DataFrame(data2_train)
    data_train = copy.copy(data2_train)
    data2_train.set_index(new_index, inplace=True)

    data2_train_fit = data2_train[0:(len(data2_train))]
    model = ARIMA(data2_train_fit, order=(4, 2, 2), freq=FREQ)

    pred_begin = pd.date_range('20180101', periods=len(data2_train), freq = FREQ)[-1]    
    arima_result = model.fit()
    forcast_vals_np = arima_result.forecast(11)[0]
    fore_new_index = pd.date_range(pred_begin, periods=len(forcast_vals_np))
    forcast_vals = pd.DataFrame(forcast_vals_np)
    forcast_vals.set_index(fore_new_index, inplace=True)
    fore_stock_forcast = pd.concat([data2_train, forcast_vals], axis=1,
                                   keys=['original', 'predicted'])  
    pred_x = np.linspace(0,241*3*11,12)
    pred_x = pred_x[0:11]
    xval = np.linspace(0,241*3*11, 241*3*11+1)
    xval = xval[0:241*3*11]
    yinter = itp.spline(pred_x,forcast_vals_np, xval)
    df_results[stock_code[stock_index]] = yinter

    a  = 1

def main():
    fr = open('StockFile.pkl', 'rb')
    # fr1 = open("left11.pkl", 'rb')
    # left_code = pickle.load(fr1)
    data1 = pickle.load(fr)
    df, stock_code = data1[0], data1[1]
    error_stock = []
    j=0
    for i in range(0,512):
        try:
            #if (stock_code[i] in left_code):
            run_main(i,df,stock_code)
            j=j+1
        except Exception as e:
            print(e)
            print("the error stock is%s"%stock_code[i])
            error_stock.append(stock_code[i])
    df_results.to_csv('out/RESULTS__%d.csv'%j)
    fw = open('out/left%d.pkl'%(512-j), 'wb')
    pickle.dump(error_stock, fw)
    return

if __name__ == "__main__":
    main()