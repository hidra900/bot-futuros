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
df = client.get_candlestick_data(symbol='MATICUSDT', interval='15m', limit=1000)

df = Parse_data(df,1000)

#df['start']=pd.to_datetime(df['start'])
#df=df[df.start>=dt.datetime.strptime('11/01/21','%m/%d/%y')]
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
#if df.RSI == '30' :

signal['signal'] = np.where(df.media_close_10 < df.media_close_55,1,0)
signal['position'] = signal['signal'].diff()



signal['signal-rsi-c'] = np.where(df.RSI.values < 30,1,0)
signal['position-rsi-c'] = signal['signal-rsi-c'].diff()
signal['signal-rsi-v'] = np.where(df.RSI.values > 70,-1,0)
signal['position-rsi-v'] = signal['signal-rsi-v'].diff()





# #################################################################
# #################################################################
# #################################################################

capital = 100000
stocks = int(1000)

positions = stocks*signal['signal']
portfolio = positions.multiply(df.close)

pos_diff = positions.diff()

cash = capital - (pos_diff.multiply(df.close).cumsum())
total = cash + portfolio

returns = total.pct_change()[1:]
returns = returns[returns != 0]

print("\n Valor total bruto de la cartera al final del periodo:", round(total.iloc[-1] ,2))

# fig= plt.figure(figsize=(20,10))
# ax1 = fig.add_subplot(121)
# ax1.set_title("Valor bruto de la cartera con " + str(capital) + " euros y " + str(stocks) + " acciones")
# total.plot(ax=ax1, lw=2.)
# ax1.plot(total[signal['position'] == 1], '^', markersize=9, color='g')
# ax1.plot(total[signal['position'] == -1], 'v', markersize=9, color='r')

# ax2 = fig.add_subplot(122)
# ax2.set_title("Frecuencia de los retornos")
# sns.histplot(returns, kde=True, ax=ax2)

# grafico = plt.figure(figsize=(20,10))
# tabla = gridspec.GridSpec(nrows=2, ncols=1, figure=grafico, height_ratios=[3,1])

# graf_sup = plt.subplot(tabla[0,0])
# graf_inf = plt.subplot(tabla[1,0])

# graf_sup.plot(df.close, label='Cierre')
# graf_sup.plot(df.close[signal['position']== 1], '^', markersize=9, color='g')
# graf_sup.plot(df.close[signal['position']== -1], 'v', markersize=9, color='r')
# graf_sup.set_title("Precio")

# graf_inf.plot(df.index, macd, 'b', label="MACD")
# graf_inf.plot(df.index, ema9, 'r--', label="Signal")
# graf_inf.bar(df.index, histograma, color=(histograma>0).map({True:'g', False:'r'}))
# graf_inf.set_title("MACD")
# plt.grid()
# plt.show()


fig, ax = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2.5, 1,1]}, figsize=(16,8))
#plt.figure(figsize=(20,10))
ax[0].bar(df.index.values, df['diff'], width=0.9, bottom=df.open, color=df.DIFF.map({True:'g', False:'r'}))
ax[0].bar(df.index.values, df['diffh'], width=0.3,bottom=df.open,color=df.DIFF.map({True:'g', False:'r'}))
ax[0].bar(df.index.values, df['diffl'], width=0.3,bottom=df.open,color=df['DIFF'].map({True:'g', False:'r'}))
ax[0].plot(df.close[signal['position']== -1], '^', markersize=9, color='b')
ax[0].plot(df.close[signal['position']== 1], 'v', markersize=9, color='k')
ax[0].plot(df.media_close_10,'b')
ax[0].plot(df.media_close_55,'orange')
ax[0].plot(df.media_close_200,'p')
ax[0].set_title('MATICUSDT')
ax[0].set_ylabel('precio')
ax[0].grid(True)


ax[1].plot(df.index, macd, 'b', label="MACD")
ax[1].plot(df.index, ema9, 'r--', label="Signal")
ax[1].bar(df.index, histograma, color=(histograma>0).map({True:'g', False:'r'}))
#ax[1].grid(True)

ax[2].plot(df.RSI)
ax[2].plot(df.index,70*np.ones(df.shape[0]),'r')
ax[2].plot(df.index,30*np.ones(df.shape[0]),'g')
ax[2].plot(df.RSI[signal['position-rsi-c']== 1], '^', markersize=9, color='b')
ax[2].plot(df.RSI[signal['position-rsi-v']== -1], 'v', markersize=9, color='k')
ax[2].set_title('RSI_14')
plt.show()

# #df=pd.read_csv('matic_15m_2021.csv')

# #df['close'].rolling(window=10,min_periods=1).mean()



# # df = client.get_candlestick_data(symbol="MATICUSDT", interval=CandlestickInterval.MIN15, startTime="1 Oct, 2021", endTime="18 Nov, 2021", limit=1000)
# # df = Parse_data(df,400)
# # df = df.iloc[:,:6]
# # df.columns = ['time','open','high','low','close','volume']
# # df = df.set_index('time')
# # df.index = pd.to_datetime(df.index, unit='ms')
# # df = df.astype(float)
  
# # #print(getminutedata('MATICUSDT'))

# # def MACD(df):
# #     df['EMA12'] = df.close.ewm(span=12).mean()
# #     df['EMA26'] = df.close.ewm(span=26).mean()
# #     df['MACD'] = df.EMA12 - df.EMA26
# #     df['signal'] = df.MACD.ewm(span=9).mean()
# #     df['histograma'] = df.MACD - df.signal
# #     print('indicador added')

# def media_movil(num,df,columns):
# 	df['media_{}_{}'.format(columns,num)]=df[columns].rolling(window=int(num),min_periods=1).mean()
# 	return df


# medias=['10','55','200']
# for m in medias:
# 	df=media_movil(m,df,'close')


# #df['start']=pd.to_datetime(df['start'])
    
# m_rap = data.ewm(span=12, adjust=False).mean()
# m_lenta = data.ewm(span=26, adjust=False).mean()

# macd = m_rap - m_lenta
# ema9 = macd.ewm(span=9, adjust=False).mean()
# histograma = macd - ema9


# # MACD(df)
# # grafico = plt.figure(figsize=(20,10))
# # tabla = gridspec.GridSpec(nrows=2, ncols=1, figure=grafico, height_ratios=[3,1])
# # graf_inf = plt.subplot(tabla[1,0])
# # graf_inf.plot(df.signal, label='signal', color='red')
# # graf_inf.plot(df.MACD, label='MACD', color='blue')
# # graf_inf.bar(df.index, df.histograma,  color=(df.histograma>0).map({True:'g', False:'r'}))
# # graf_inf.set_title("MACD")
# # plt.grid()
# # plt.show()

# df['diff']=df.close.diff(periods=1)
# df.dropna(inplace=True)
# df['sub']=df['diff'][df['diff']>0]
# df['baj']=abs(df['diff'][df['diff']<=0])
# df.fillna(value=0,inplace=True)
# df['media_sub_14']=df['sub'].rolling(window=14).mean()
# df['media_baj_14']=df['baj'].rolling(window=14).mean()
# df['RSI']=100-(100/(1+(df.media_sub_14/df.media_baj_14)))
# df.dropna(inplace=True)
# df.drop(columns=['diff','sub','baj','media_sub_14','media_baj_14'],inplace=True)

# df['diff']=df.close-df.open
# df['diffh']=df.high-df.open
# df['diffl']=df.low-df.open
# df['DIFF']=df['diff']>0

# fig, ax = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [2.5, 1,1]}, figsize=(16,8))


# ax[0].bar(df.index.values, df['diff'], width=0.9, bottom=df.open, color=df.DIFF.map({True:'g', False:'r'}))
# ax[0].bar(df.index.values, df['diffh'], width=0.3,bottom=df.open,color=df.DIFF.map({True:'g', False:'r'}))
# ax[0].bar(df.index.values, df['diffl'], width=0.3,bottom=df.open,color=df['DIFF'].map({True:'g', False:'r'}))
# #ax[0].bar(df.index.values, df['volume'], width=0.3, color=df.DIFF.map({True:'g', False:'r'}))
# ax[0].plot(df.media_close_10,'b')
# ax[0].plot(df.media_close_55,'orange')
# ax[0].plot(df.media_close_200,'p')
# ax[0].set_title('MATICUSDT')
# ax[0].set_ylabel('Price')
# #ax[0].set_title('volume')
# ax[0].legend(['media_movil_10','media_movil_55','media_movil_200'])
# ax[0].grid(True)


# ax[1].plot(df.RSI)
# ax[1].plot(df.index,70*np.ones(df.shape[0]),'r')
# ax[1].plot(df.index,30*np.ones(df.shape[0]),'g')
# ax[1].set_title('RSI_14')



# ax[2].plot(df.signal, label='signal', color='red')
# ax[2].plot(df.MACD, label='MACD', color='blue')
# ax[2].bar(df.index, df.histograma,  color=(df.histograma>0).map({True:'g', False:'r'}))
# ax[2].set_title('MACD')
# plt.show()