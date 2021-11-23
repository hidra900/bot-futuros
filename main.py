from config import *
from config_bot import *
from bot import *



Bot = Cripto_Bot(api_key=api_key, api_secret=secret_key, cripto=CRIPTO, ref=REF, period=PERIOD, leverage=LEVERAGE,
                sma_f=SMA_F,sma_s=SMA_S, side=SIDE, capital=CAPITAL)

Bot.run()

