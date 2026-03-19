import pandas as pd
import numpy as np
df=pd.read_csv("CE_calc.csv")
  
df_trades = pd.DataFrame(columns=["begin_date","end_date","total_return","direction"])
number=0
i=0
last=0
df_trades.loc[0,"total_return"]=0
for row in range(len(df)):
    t = df.loc[row, "Signal"]
    if last != 0:
        df_trades.loc[i, "total_return"] += df.loc[row, "Returns"] * last
    
    if t==1:
        number+=1
        df_trades.loc[i,"direction"]="long"
        df_trades.loc[i,"total_return"]=0
        df_trades.loc[i,"begin_date"]=df.loc[row+1,"Date"]
        last=1
        
                
     

    elif t==-1:
        if last != 0:
            df_trades.loc[i, "end_date"] = df.loc[row, "Date"]
            i += 1
        number+=1
        last=0
        #df_trades.loc[i,"direction"]="short"
        #df_trades.loc[i,"begin_date"]=df.loc[row+1,"Date"]
        #df_trades.loc[i, "total_return"] = 0
        #last=-1
    
        
    #else:    
        #df_trades.loc[i,"total_return"]+=df.loc[row,"Returns"]*last

if last != 0:
    df_trades.loc[i, "end_date"] = df.loc[len(df)-1, "Date"]
    
df_trades.to_csv("trades_ohnefilter_longonly.csv", index=False)

################################################################################
################################################################################



df_trades["total_return"] = df_trades["total_return"] / 100
# Cumulative equity per trade
df_trades["cum_equity"] = (1 + df_trades["total_return"]).cumprod()

# Maximum Drawdown
df_trades["cummax"] = df_trades["cum_equity"].cummax()
df_trades["drawdown"] = df_trades["cum_equity"] / df_trades["cummax"] - 1
max_drawdown = df_trades["drawdown"].min()

# Sharpe ratio
sharpe_trade = df_trades["total_return"].mean() / df_trades["total_return"].std()

# Win/loss rate
win_rate = (df_trades["total_return"] > 0).mean()
loss_rate = (df_trades["total_return"] < 0).mean()
total_return = df_trades["cum_equity"].iloc[-1] - 1

# --- Step 7: Print results ---
print("===== Trade Metrics =====")
print("Number of trades:", len(df_trades))
print("Win rate:", round(win_rate, 4))
print("Loss rate:", round(loss_rate, 4))
print("Sharpe ratio:", round(sharpe_trade, 4))
print("Total return:", round(total_return * 100, 2), "%")
print("Max Drawdown:", round(max_drawdown * 100, 2), "%")
