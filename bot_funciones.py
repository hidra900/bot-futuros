from binance_f import *
from config import *
from binance_f import RequestClient
from binance_f.constant.test import *
from binance_f.base.printobject import *
from binance_f.model.constant import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime as dt
import pandas_datareader.data as wb
import seaborn as sns
import math
import mplfinance as mpf


def Parse_data(result, limit):
    """

    :param result:
    :param limit:
    :return:
    """
    data = []
    for i in range(limit):
        vela = []
        vela.append(result[i].openTime)
        vela.append(result[i].open)
        vela.append(result[i].high)
        vela.append(result[i].low)
        vela.append(result[i].close)
        vela.append(result[i].volume)
        data.append(vela)
    df = pd.DataFrame(data)
    col_names = ['time','open','high','low','close','volume']
    df.columns = col_names
    for col in col_names:
        df[col] = df[col].astype(float)
    df['start'] = pd.to_datetime(df['time'] * 1000000)

    return df

def Get_Capital(result, ref):
    for i in range(len(result.assets)):
        if result.assets[i].asset == ref:
            return result.assets[i].marginBalance



def Get_Exchange_filters(result, S):
    for i in range(len(result.symbols)):
        if result.symbols[i].symbol == S:
            minQty = float(result.symbols[i].filters[2].get('minQty'))
            stepSize = float(result.symbols[i].filters[2].get('stepSize'))
            maxQty = float(result.symbols[i].filters[2].get('maxQty'))
            return minQty, stepSize, maxQty

def Calculate_max_Decimal_Qty(stepSize):
    max_decimal_quantity=0
    a = 10
    while stepSize*a<1:
      a = a*10**max_decimal_quantity
      max_decimal_quantity += 1
    return max_decimal_quantity



def Crossover(MF, MS):
    if (MF[0] < MS[0] and MF[1] >= MS[1]):
        return True
    else:
        return False

def Calculate_Qty(price, money, minQty, maxQty, maxDeciamlQty):

    Q = money / price
    if (Q < minQty or Q > maxQty):
        return False
    Q = np.round(Q, maxDeciamlQty)
    return Q





#======= indicadores =========
client = RequestClient(api_key=api_key, secret_key=secret_key)
df = client.get_candlestick_data(symbol='MATICUSDT', interval='15m', limit=200)
#df = pd.read_csv('matic_15m_1mes.csv')
df = Parse_data(df,200)


#df.index =  pd.to_datetime(df.index)

df['start']=pd.to_datetime(df['start'])

df[df.start>=dt.datetime.strptime('10/29/21','%m/%d/%y')]

df = df[df.start>=dt.datetime.strptime('10/29/21','%m/%d/%y')]
# print(df_temp)
#df.set_index('start',inplace=True)

df['close'].rolling(window=10,min_periods=1).mean()

df['diff']=df.close.diff(periods=1)
df.dropna(inplace=True)
df['sub']=df['diff'][df['diff']>0]
df['baj']=abs(df['diff'][df['diff']<=0])
df.fillna(value=0,inplace=True)
df['media_sub_14']=df['sub'].rolling(window=14).mean()
df['media_baj_14']=df['baj'].rolling(window=14).mean()
df['RSI']=100-(100/(1+(df.media_sub_14/df.media_baj_14)))
df.dropna(inplace=True)
df.drop(columns=['diff','sub','baj','media_sub_14','media_baj_14'],inplace=True)

df['diff']=df.close-df.open
df['diffh']=df.high-df.open
df['diffl']=df.low-df.open
df['DIFF']=df['diff']>0


# #################################################################

                    #Squeeze momentum

length = 20
mult = 2
length_KC = 20
mult_KC = 1.5

# parameter setup
length = 20
mult = 2
length_KC = 20
mult_KC = 1.5

# calculate BB
m_avg = df['close'].rolling(window=length).mean()
m_std = df['close'].rolling(window=length).std(ddof=0)
df['upper_BB'] = m_avg + mult * m_std
df['lower_BB'] = m_avg - mult * m_std

# calculate true range
df['tr0'] = abs(df["high"] - df["low"])
df['tr1'] = abs(df["high"] - df["close"].shift())
df['tr2'] = abs(df["low"] - df["close"].shift())
df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

# calculate KC
range_ma = df['tr'].rolling(window=length_KC).mean()
df['upper_KC'] = m_avg + range_ma * mult_KC
df['lower_KC'] = m_avg - range_ma * mult_KC

# calculate bar value
highest = df['high'].rolling(window = length_KC).max()
lowest = df['low'].rolling(window = length_KC).min()
m1 = (highest + lowest)/2
df['value'] = (df['close'] - (m1 + m_avg)/2)
fit_y = np.array(range(0,length_KC))
df['value'] = df['value'].rolling(window = length_KC).apply(lambda x: 
                          np.polyfit(fit_y, x, 1)[0] * (length_KC-1) + 
                          np.polyfit(fit_y, x, 1)[1], raw=True)

# check for 'squeeze'
df['squeeze_on'] = (df['lower_BB'] > df['lower_KC']) & (df['upper_BB'] < df['upper_KC'])
df['squeeze_off'] = (df['lower_BB'] < df['lower_KC']) & (df['upper_BB'] > df['upper_KC'])

# buying window for long position:
# 1. black cross becomes gray (the squeeze is released)
long_cond1 = (df['squeeze_off'].iloc[-2] == False) & (df['squeeze_off'].iloc[-1] == True) 
# 2. bar value is positive => the bar is light green k
long_cond2 = df['value'].iloc[-1] > 0
enter_long = long_cond1 and long_cond2

# buying window for short position:
# 1. black cross becomes gray (the squeeze is released)
short_cond1 = (df['squeeze_off'].iloc[-2] == False) & (df['squeeze_off'].iloc[-1] == True) 
# 2. bar value is negative => the bar is light red 
short_cond2 = df['value'].iloc[-1] < 0
enter_short = short_cond1 and short_cond2



# to make the visualization better by only taking the last 100 rows of data
#df = df.iloc[-1000:]

# extract only ['Open', 'High', 'Close', 'Low'] from df
ohcl = df[['open', 'high', 'close', 'low']]

# add colors for the 'value bar'
colors = []
for ind, val in enumerate(df['value']):
  if val >= 0:
    color = 'green'
    if val > df['value'].iloc[ind-1]:
      color = 'lime'
  else:
    color = 'maroon'
    if val < df['value'].iloc[ind-1]:
      color='red'
  colors.append(color)
  
#add 2 subplots: 1. bars, 2. crosses
# apds = [mpf.make_addplot(df['value'], panel=1, type='bar', color=colors, alpha=0.8, secondary_y=False),
#        mpf.make_addplot([0] * len(df), panel=1, type='scatter', marker='x', markersize=50, color=['gray' if s else 'black' for s in df['squeeze_off']], secondary_y=False)]

# #plot ohcl with subplots
# fig, axes = mpf.plot(ohcl, 
#               volume_panel = 2,
#               figratio=(2,1),
#               figscale=1, 
#               type='candle', 
#               addplot=apds,
#               returnfig=True)
df = df.dropna()

def minimosLocalesNegativos(arr):
# lista para almacenar los minimos locales
    minLocal = []
# Chequemos si el primer elemento es un minimo local
    if arr[0] < 0 and arr[0] < arr[1]:
        minLocal.append(0)
       
# iterando sobre todos los siguientes elementos, excepto el último.
    for i in range(1, len(arr) - 1):
        if (arr[i-1] > arr[i] <= arr[i + 1]) and (arr[i] < 0):
            minLocal.append(arr[i])
# Chequemos si el ultimo elemento es un minimo local
        if (arr[-1] < arr[-2]) and arr[-1] < 0:
            minLocal.append(arr[-1])
# el output es una lista con los mínimos locales
    return minLocal
df1 = [0.1, 0.2, 0.5, 0.6, -0.15, -0.945, -0.945, -0.3, 0.1, 0.3, 0.4, 0.5, -0.14, -0.82, -0.95, -0.87, 0.1, 0.3, 0.4, 0.5, -0.16, -0.62, -0.54, -0.15]

print(minimosLocalesNegativos(df1))

# minimo = []
# for i in df['value']:
#     minimo.append(i)
# print(minimosLocalesNegativos(minimo))


# print(minimo)
# print(minimosLocalesNegativos(minimo1))

#Si lo que buscas son los índices solo hay que reemplazar los append(arr[i]) por append(i)
# minimo = []
# maximo = []
# for i in df['value']:
#     if i < 0:
#         minimo.append(i)
#     continue
#     if i > 0:
#         break
# print(minimo)




# #################################################################

#print(df.RSI)

for m in df.RSI:
    if m <= 30:
        compra = "compra"
        

    elif m >=70:
        venta="venta"
        

def media_movil(num,df,columns):
    df['media_{}_{}'.format(columns,num)]=df[columns].rolling(window=int(num),min_periods=1).mean()
    return df


medias=['10','55','200']
for m in medias:
    df=media_movil(m,df,'close')

m_rap = df.close.ewm(span=12, adjust=False).mean()
m_lenta = df.close.ewm(span=26, adjust=False).mean()

macd = m_rap - m_lenta
ema9 = macd.ewm(span=9, adjust=False).mean()
histograma = macd - ema9


signal = pd.DataFrame(index=df.index)


signal['signal'] = np.where(df.media_close_10 < df.media_close_55,1,0)
signal['position'] = signal['signal'].diff()


signal['signal-rsi-c'] = np.where(df.RSI.values < 30,1,0)
signal['position-rsi-c'] = signal['signal-rsi-c'].diff()
signal['signal-rsi-v'] = np.where(df.RSI.values > 70,-1,0)
signal['position-rsi-v'] = signal['signal-rsi-v'].diff()

signal['signal-squeeze'] = np.where(df['value'] > 0,1,0)
signal['signal-squeeze-c'] = signal['signal-squeeze'].diff()



# #################################################################

capital = 1000
stocks = int(10)

positions = stocks*signal['signal']
portfolio = positions.multiply(df.close)

pos_diff = positions.diff()

cash = capital - (pos_diff.multiply(df.close).cumsum())
total = cash + portfolio


returns = total.pct_change()[1:]
returns = returns[returns != 0]

#print("\n Valor total bruto de la cartera al final del periodo:", round(total.iloc[-1] ,2))



# fig, ax = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2.5, 1,1]}, figsize=(16,8))
# #plt.figure(figsize=(20,10))
# ax[0].bar(df.index.values, df['diff'], width=0.9, bottom=df.open, color=df.DIFF.map({True:'g', False:'r'}))
# ax[0].bar(df.index.values, df['diffh'], width=0.3,bottom=df.open,color=df.DIFF.map({True:'g', False:'r'}))
# ax[0].bar(df.index.values, df['diffl'], width=0.3,bottom=df.open,color=df['DIFF'].map({True:'g', False:'r'}))
# ax[0].plot(df.close[signal['position']== 1], '^', markersize=9, color='b')
# ax[0].plot(df.close[signal['position']== -1], 'v', markersize=9, color='k')
# ax[0].plot(df.media_close_10,'b')
# ax[0].plot(df.media_close_55,'orange')
# ax[0].plot(df.media_close_200,'p')
# ax[0].set_title('MATICUSDT')
# ax[0].set_ylabel('precio')
# ax[0].grid(True)


# # ax[1].plot(df.index, macd, 'b', label="MACD")
# # ax[1].plot(df.index, ema9, 'r--', label="Signal")
# # ax[1].bar(df.index, histograma, color=(histograma>0).map({True:'g', False:'r'}))
# # ax[1].grid(True)


# #ax[1].plot(df.index, macd, 'b', label="MACD")
# # ax[1].plot(df['value'][signal['signal-squeeze-c'] < 0], '^', markersize=9, color='b')
# ax[1].bar(df.index, df['value'], color=(df['value'] > 0).map({True:'g', False:'r'}))
# ax[1].grid(True)

# ax[2].plot(df.RSI)
# ax[2].plot(df.index,70*np.ones(df.shape[0]),'r')
# ax[2].plot(df.index,30*np.ones(df.shape[0]),'g')
# ax[2].plot(df.RSI[signal['position-rsi-c']== 1], '^', markersize=9, color='b')
# ax[2].plot(df.RSI[signal['position-rsi-v']== -1], 'v', markersize=9, color='k')
# ax[2].set_title('RSI_14')
# plt.show()

# prev_df = list()
# for candlestick in res:
#   row = dict()
#   row['close'] = candlestick.close
#   row['closeTime'] = candlestick.closeTime
#   row['high'] = candlestick.high
#   row['low'] = candlestick.low
#   row['open'] = candlestick.open
#   row['openTime'] = candlestick.openTime
#   row['volume'] = candlestick.volume
#   prev_df.append(row)