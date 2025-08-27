import yfinance as yf
import matplotlib.pyplot as plt
data=yf.dowmload('msft',start='2025-01-01)
                 
data.head(10)
