from bot_funciones import*
import time
import datetime as dt
import pytz


class Cripto_Bot():

	def __init__(self,api_key, api_secret, cripto, ref, period, leverage, sma_f,sma_s, side, capital):
        # crear clente

		self.client = RequestClient(api_key=api_key, secret_key=secret_key)
		
		#variables funcionales
		#filtros
		try:
			self.client.change_position_mode(dualSidePosition=True)
		except:
			pass

		#parametros

		self.cripto = cripto
		self.ref = ref
		self.exchange = self.cripto+self.ref
		self.side = side
		self.single_operation_capital = capital
		self.leverage = leverage
		self.SMA_F = sma_f
		self.SMA_S = sma_s
		self.period = period

		
		try:
			self.client.change_initial_leverage(symbol=self.cripto+self.ref, leverage=self.leverage)
		except:
			self.RUN = False
			print('Error leverage')
        # Filtros

		result = self.client.get_exchange_information()
		self.minQty, self.stepSize, self.maxQty = Get_Exchange_filters(result, self.exchange)

		self.maxDeciamlQty = Calculate_max_Decimal_Qty(self.stepSize)

		self.capital = Get_Capital(self.client.get_account_information(), self.ref)

        # Variables logisticas

		self.df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume','start','SMA_F','SMA_S'])

		self.buysignal = None
		self.sellsignal = None


		self.quantity = None
		self.open = False


		self.RUN = True

	def Last_data(self):
		if self.df.shape[0] == 0:
			candles = self.client.get_candlestick_data(symbol=self.exchange, interval=self.period,limit=self.SMA_S + 1)
			self.df = Parse_data(candles, limit=self.SMA_S+1)

		else:

			candles = self.client.get_candlestick_data(symbol=self.exchange, interval=self.period,limit=self.SMA_S + 1)
			df_temp = Parse_data(candles, limit=1)
			self.df = self.df.append(df_temp, ignore_index=True)
			self.df.drop(index=0, inplace=True)
			self.df.index = list(range(self.SMA_S + 1))


		self.df['SMA_F'] = self.df['close'].rolling(self.SMA_F).mean()
		self.df['SMA_S'] = self.df['close'].rolling(self.SMA_S).mean()



		if self.side == 'LONG':
			self.buysignal = Crossover(self.df.SMA_F.values[-2:], self.df.SMA_S.values[-2:])
			self.sellsignal = Crossover(self.df.SMA_S.values[-2:], self.df.SMA_F.values[-2:])
		else:

			self.buysignal = Crossover(self.df.SMA_S.values[-2:], self.df.SMA_F.values[-2:])
			self.sellsignal = Crossover(self.df.SMA_F.values[-2:], self.df.SMA_S.values[-2:])


	def Orden(self):
		Qty = Calculate_Qty(price, self.single_operation_capital*self.leverage, self.minQty, self.maxQty, self.maxDeciamlQty)
		if not Qty:
			self.RUN = False
		if self.side == 'LONG':
			if side == 'BUY':
				self.quantity = Qty
				self.open = True
			else:
				self.open = False
		else:
			if side == 'SELL':
				self.quantity = Qty
				self.open = True
			else:
				self.open = False

		self.client.post_order(symbol=self.exchange, side=side, ordertype='MARKET', quantity=self.quantity,
                               positionSide=self.side)

	def Single_Operation(self):
		self.capital = Get_Capital(self.client.get_account_information(), self.ref)
		if self.capital <= self.single_operation_capital:
			print('Dinero no suficiente')
			self.RUN = False
        # actualizar datos
		self.Last_data()

        # precio actual
		price = self.client.get_symbol_price_ticker(symbol=self.cripto + self.ref)[0].price

		if self.open:
			if self.sellsignal:
				if self.side == 'LONG':
					side = 'SELL'
					try:
						self.Order(side=side,price=price)
					except Exception as e:
						print(e)

				else:
					side = 'BUY'
					try:
						self.Order(side=side,price=price)
						self.H_df.operacion.iloc[-1] = 'BUY'
					except Exception as e:
						print(e)
		else:
			if self.buysignal:
				if self.side == 'LONG':
					side = 'BUY'
					try:
						self.Order(side=side,price=price)
					except Exception as e:
						print(e)
				else:
					side = 'SELL'
					try:
						self.Order(side=side,price=price)

					except Exception as e:
						print(e)

	def run(self):
		if 'm' in self.period:
			if len(self.period) == 2:
				step = int(self.period[0])
			else:
				step = int(self.period[:2])
		elif self.period == '1h':
			step = 60
		else:
			print('interval error')
			return
		self.Last_data()
		START = self.df.start.iloc[-1] + dt.timedelta(minutes=step)
		print(START)
		while dt.datetime.now(dt.timezone.utc) < pytz.UTC.localize(START):
			time.sleep(1)
			pass
			print('Strarting Bot...\n')
		time.sleep(3)  # para ser seguros de encontrar los datos de la velas siguente
		print('Bot started')
		while self.RUN:
			temp = time.time()
			self.Single_Operation()
			retraso = time.time() - temp
			time.sleep(60 * step - retraso)






