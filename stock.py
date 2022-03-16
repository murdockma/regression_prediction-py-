from datetime import datetime
import requests

def stock_predictor(stock, years):
    
    cryp_ = requests.get("https://api.coingecko.com/api/v3/coins/"+str(stock)+"/market_chart/range?vs_currency=usd&from=1231039220&to=1646121661").json()
    utc_ls = []
    val_ls = []
    for key, val in cryp_['prices']:
    #     print(key)
        utc_ls.append(key)
        val_ls.append(val)

    df = pd.DataFrame({'ds':utc_ls,'y':val_ls})

    df['ds'] = df['ds'].apply(lambda x: datetime.utcfromtimestamp(int(x)/1000).strftime('%Y-%m-%d'))

    stock_prophet_df = Prophet()

    stock_prophet_df.fit(df)

    stock_future_df = stock_prophet_df.make_future_dataframe(periods=365*int(years), freq = 'D')

    stock_forecast = stock_prophet_df.predict(stock_future_df)
    
    stock_plot = stock_prophet_df.plot(stock_forecast)
    
    stock_components = stock_prophet_df.plot_components(stock_forecast)