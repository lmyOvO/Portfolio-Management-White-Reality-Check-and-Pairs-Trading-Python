#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:57:31 2021
"""


#import needed modules
from datetime import datetime
#from pandas_datareader import data
import pandas as pd
import numpy as np
from numpy import log, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
#import pprint
import sqlite3 as db
import detrendPrice

pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
import itertools as it
    

#Loop Pairs Trading backtest
def backtest(symbList, whichfirst):
    global msg
    msg = ""
    
    entryZscore = 1
    exitZscore = -0
    window =  7
    regression = 1
    residuals_model = 0
    #time period
    start_date = '2011-01-01' 
    end_date = '2021-01-01'
    
    #build dataframe
    if whichfirst == "yfirst":
        try:
            y = pdr.get_data_yahoo(symbList[0], start=start_date, end=end_date)
            y.to_csv(symbList[0] + '.csv', header = True, index=True, encoding='utf-8')
            x = pdr.get_data_yahoo(symbList[1], start=start_date, end=end_date)
            x.to_csv(symbList[1] + '.csv', header = True, index=True, encoding='utf-8')
        except Exception:
            msg = "yahoo problem"
            y = pd.DataFrame()
            x = pd.DataFrame()
            return 0
    else:
        try:
            y = pdr.get_data_yahoo(symbList[1], start=start_date, end=end_date)
            y.to_csv(symbList[1] + '.csv', header = True, index=True, encoding='utf-8')
            x = pdr.get_data_yahoo(symbList[0], start=start_date, end=end_date)
            x.to_csv(symbList[0] + '.csv', header = True, index=True, encoding='utf-8')
        except Exception:           
            msg = "yahoo problem"
            y = pd.DataFrame()
            x = pd.DataFrame()
            return 0

    if whichfirst == "yfirst":
        y = pd.read_csv(symbList[0] + '.csv', parse_dates=['Date'])
        y = y.sort_values(by='Date')
        y.set_index('Date', inplace = True)
        x = pd.read_csv(symbList[1] + '.csv', parse_dates=['Date'])
        x = x.sort_values(by='Date')
        x.set_index('Date', inplace = True)
    else:
        y = pd.read_csv(symbList[1] + '.csv', parse_dates=['Date'])
        y = y.sort_values(by='Date')
        y.set_index('Date', inplace = True)
        x = pd.read_csv(symbList[0] + '.csv', parse_dates=['Date'])
        x = x.sort_values(by='Date')
        x.set_index('Date', inplace = True)
 
     
    #doing an inner join to make sure dates coincide and there are no NaNs
    #inner join requires distinct column names
    y.rename(columns={'Open':'y_Open','High':'y_High','Low':'y_Low','Close':'y_Close','Adj Close':'y_Adj_Close','Volume':'y_Volume'}, inplace=True) 
    x.rename(columns={'Open':'x_Open','High':'x_High','Low':'x_Low','Close':'x_Close','Adj Close':'x_Adj_Close','Volume':'x_Volume'}, inplace=True) 
    df1 = pd.merge(x, y, left_index=True, right_index=True, how='inner') #inner join
    
    '''
    plt.plot(df1.y_Adj_Close,label=symbList[0])
    plt.plot(df1.x_Adj_Close,label=symbList[1])
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
    sns.jointplot(df1.y_Adj_Close, df1.x_Adj_Close ,color='b')
    plt.show()
    '''
    
    #get rid of extra columns but keep the date index
    df1.drop(['x_Open', 'x_High','x_Low','x_Close','x_Volume','y_Open', 'y_High','y_Low','y_Close','y_Volume'], axis=1, inplace=True)
    df1.rename(columns={'y_Adj_Close':'y','x_Adj_Close':'x'}, inplace=True) 
    df1 = df1.assign(TIME = pd.Series(np.arange(df1.shape[0])).values) 
    
    #repeat for detrended prices
    df1 = df1.assign(x_DETREND =  detrendPrice.detrendPrice(df1.x).values)
    df1 = df1.assign(y_DETREND =  detrendPrice.detrendPrice(df1.y).values)
    df1 = df1.assign(TIME = pd.Series(np.arange(df1.shape[0])).values) 

    #find the hedge ratio and the spread
    #regress the y variable against the x variable
    #the slope of the rolling linear univariate regression=the rolling hedge ratio
    
    window_hr_reg = 58 #smallest window for regression when using y_hat
    
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1["y"].values
    x_ = df1[['x']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window_hr_reg, len(df1)):
        y = y_[(n - window_hr_reg):n]
        X = x_[(n - window_hr_reg):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    if residuals_model:
        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                 myList.append(0)
            else:
                myList.append(b[e][0])
        df1["rolling_hedge_ratio"] = myList
    else:
        df1 = df1.assign(rolling_hedge_ratio = pd.Series(np.ones(df1.shape[0])).values)
        
    #repeat for detrended prices
    #use for White Reality Check
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1["y_DETREND"].values
    x_ = df1[['x_DETREND']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window_hr_reg, len(df1)):
        y = y_[(n - window_hr_reg):n]
        X = x_[(n - window_hr_reg):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept

    if residuals_model:
        myList = []
        for e in range(len(b)):
            if e < window_hr_reg:
                 myList.append(0)
            else:
                 myList.append(b[e][0])
        df1["rolling_hedge_ratio_DETREND"] = myList
    else:
        df1 = df1.assign(rolling_hedge_ratio_DETREND = pd.Series(np.ones(df1.shape[0])).values)
            
    #calculate the spread
    if residuals_model == 1:
        df1['spread'] = df1.y - df1.rolling_hedge_ratio*df1.x
    else:
        df1['spread'] = log(df1.y) - log(df1.x)
    
    #rolling regression instead of moving average
    a = np.array([np.nan] * len(df1))
    b = [np.nan] * len(df1)  # If betas required.
    y_ = df1['spread'].values
    x_ = df1[['TIME']].assign(constant=0).values #if constant=0, intercept is forced to zero; if constant=1 intercept is as usual
    for n in range(window, len(df1)):
        y = y_[(n - window):n]
        X = x_[(n - window):n]
        # betas = Inverse(X'.X).X'.y
        betas = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = betas.dot(x_[n, :])
        a[n] = y_hat
        b[n] = betas.tolist()  # If betas required. b[n][0] is the slope, b[n][1] is the intercept
    
    df1 = df1.assign(y_hat = pd.Series(a).values)
        
    if regression == 1:
        mean = df1['y_hat']
    else:
        mean = df1['spread'].rolling(window=window).mean()
    
    #calculate the zScore indicator
    df1 = df1.assign(meanSpread = pd.Series(a).values)
    stdSpread = df1.spread.rolling(window=window).std()
    df1['zScore'] = (df1.spread-mean)/stdSpread
    
    """
    plt.plot(df1.spread)
    plt.show()
    df1['zScore'].plot()
    plt.show()
    """
    
    #set up num units long             
    df1['long entry'] = ((df1.zScore < - entryZscore))
    df1['long exit'] = ((df1.zScore > - exitZscore)) 
    df1['num units long'] = np.nan 
    df1.loc[df1['long entry'],'num units long'] = 1 
    df1.loc[df1['long exit'],'num units long'] = 0 
    df1.iat[0,df1.columns.get_loc("num units long")]= 0
    
    df1['num units long'] = df1['num units long'].fillna(method='pad') 
    
    #set up num units short 
    df1['short entry'] = ((df1.zScore >  entryZscore))
    df1['short exit'] = ((df1.zScore < exitZscore))
    df1['num units short'] = np.nan
    df1.loc[df1['short entry'],'num units short'] = -1 
    df1.loc[df1['short exit'],'num units short'] = 0
    df1.iat[0,df1.columns.get_loc("num units short")]= 0
    df1['num units short'] = df1['num units short'].fillna(method='pad')
###############################################################################################################################
    df1['numUnits'] = df1['num units long'] + df1['num units short']
    
    #positions_ = dollar capital allocation in each ETF
    df1["positions_x"] =-1*df1["rolling_hedge_ratio"]*df1["x"]*df1["numUnits"]
    df1["positions_y"] =df1["y"]*df1["numUnits"]
    df1["price_change_x"] = df1["x"] - df1["x"].shift(1)
    df1["price_change_y"] = df1["y"] - df1["y"].shift(1)
    df1["pnl_x"] = df1["price_change_x"]*df1["positions_x"].shift(1)/df1["x"].shift(1)
    df1["pnl_y"] = df1["price_change_y"]*df1["positions_y"].shift(1)/df1["y"].shift(1)
    df1["pnl"] = df1["pnl_x"] + df1["pnl_y"] 
    df1["portfolio_cost"] = np.abs(df1["positions_x"])+np.abs(df1["positions_y"])
    df1["port_rets"]= df1["pnl"]/df1["portfolio_cost"].shift(1)
    df1["port_rets"].fillna(0, inplace=True)

    #repeat for detrended prices
    #use for White Reality Check
    df1["positions_x_DETREND"] =-1*df1["rolling_hedge_ratio_DETREND"]*df1["x_DETREND"]*df1["numUnits"]
    df1["positions_y_DETREND"] =df1["y_DETREND"]*df1["numUnits"]
    df1["price_change_x_DETREND"] = df1["x_DETREND"] - df1["x_DETREND"].shift(1)
    df1["price_change_y_DETREND"] = df1["y"] - df1["y"].shift(1)
    df1["pnl_x_DETREND"] = df1["price_change_x_DETREND"]*df1["positions_x_DETREND"].shift(1)/df1["x_DETREND"].shift(1)
    df1["pnl_y_DETREND"] = df1["price_change_y_DETREND"]*df1["positions_y_DETREND"].shift(1)/df1["y_DETREND"].shift(1)
    df1["pnl_DETREND"] = df1["pnl_x_DETREND"] + df1["pnl_y_DETREND"] 
    df1["portfolio_cost_DETREND"] = np.abs(df1["positions_x_DETREND"])+np.abs(df1["positions_y_DETREND"])
    df1["port_rets_DETREND"]= df1["pnl_DETREND"]/df1["portfolio_cost_DETREND"].shift(1)
    df1["port_rets_DETREND"].fillna(0, inplace=True)

    df1 = df1.assign(I =np.cumprod(1+df1['port_rets'])) #this is good for pct return or log return
    df1.iat[0,df1.columns.get_loc('I')]= 1
    
    start_val = 1
    end_val = df1['I'].iat[-1]
    
    start_date = df1.iloc[0].name
    end_date = df1.iloc[-1].name
    days = (end_date - start_date).days
    
    #Total annual return
    TotaAnnReturn = (end_val-start_val)/start_val/(days/360)
    TotaAnnReturn_trading = (end_val-start_val)/start_val/(days/252)
    
    
    try:
        CAGRdbl_trading = round(((float(end_val) / float(start_val)) ** (1/(days/252.0))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
    except ZeroDivisionError:
        CAGRdbl_trading = 0
        
    try:
        CAGRdbl = round(((float(end_val) / float(start_val)) ** (1/(days/360))).real - 1,4) #when raised to an exponent I am getting a complex number, I need only the real part
    except ZeroDivisionError:
        CAGRdbl = 0

    try:
        sharpe =  (df1['port_rets'].mean()/ (df1['port_rets'].std()) * np.sqrt(252))
    except ZeroDivisionError:
        sharpe = 0.0
    #White's Reality Check 
    #returns must be detrended by subtracting the average daily return of the benchmark
    df1['port_rets_DETREND'].dropna(inplace=True)
    arr = np.array(df1['port_rets_DETREND'].values)
    alpha = .05*100 #significance alpha
    reps = 5000 #how many bootstrapings, 50000 limit if you have 8GB RAM
        
    percentile = 100-alpha
    ave = np.average(arr) #arithmetic mean

    print("average return %f" %ave)
 
    #ave = ms.gmean(arr) #geometric mean

    centered_arr = arr-ave
    n = len(centered_arr)
    #constructs 50000 alternative return histories and calculates their theoretical averages
    xb = np.random.choice(centered_arr, (n, reps), replace=True)
    mb = xb.mean(axis=0) #arithmetic mean
    #mb = ms.gmean(mb, axis=0) #geometric mean

    #sorts the 50000 averages
    mb.sort()
    #calculates the 95% conficence interval (two tails) threshold for the theoretical averages
    print(np.percentile(mb, [2.5, 97.5])) 
    threshold = np.percentile(mb, [percentile])[0]
    if ave > threshold:
        print("Reject Ho = The population distribution of rule returns has an expected value of zero or less (because p_value is small enough)")
    else:
        print("Do not reject Ho = The population distribution of rule returns has an expected value of zero or less (because p_value is not small enough)")

    #count will be the items i that are smaller than ave
    count_vals = 0
    for i in mb:
        count_vals += 1 
        if i > ave:
            break
    #p is based on the count that are larger than ave so 1-count is needed:
    p = 1-count_vals/len(mb)
    #Sharpe Ratio set to be at least larger than 0.5
    if sharpe > .5:
        #p_value set to be smaller than 0.11
        if p < 0.1: 
            if whichfirst == "yfirst":
                title=symbList[0]+"."+symbList[1]
                ylabel = symbList[0]
                xlabel = symbList[1]
            else:
                title=symbList[1]+"."+symbList[0]
                ylabel = symbList[1]
                xlabel = symbList[0]

        # print result metrics
        print("ETF PAIE IS:",xlabel,ylabel)
        print ("TotaAnnReturn = %f" %(TotaAnnReturn*100))
        print ("CAGR = %f" %(CAGRdbl*100))
        print ("Sharpe Ratio = %f" %(round(sharpe,2)))   
        print("p_value:")
        print(p)
        sharperatio=round(sharpe,2) #sharpe ratio 
        #plot equity curve
        plt.plot(df1['I'])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        #plt.show()  #if you show() it won't savefig()!
        plt.savefig(r'Results\%s.png' %(title))
        plt.close()
 
    return 1,p,sharperatio 


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################  

#MAIN 
    
#set the database file path we wish to connect to
#this will obviously be unique to wherever you created
#the SQLite database on your local system
#database = 'C:\sqlite\PythonData.db'
#Link the data to the SQLitedatabase on the local system
#Check and Change the database file path!!
database = '/Users/jenniferwang/Desktop/aps1051/S.9 D PART 3 (HOMEWORK)/SESSION 9 PART 3 (HOMEWORK)/HOMEWORK ON PAIRS TRADING/RegressionChannelTwoAssetTrading/RegressionChannelPairsTrading/sqlite/PythonData.db'  #try also 'C:\\sqlite\\PythonData.db' if you get a 'no such table' error.

 
#this is the SQL statement containing the information
#regarding which tickers we want to pull from the database
#check the SQLitedatabase for the tickers of the interesting area 
#then change the ticker's name to activate the program
#sql = 'SELECT Ticker FROM etftable WHERE "Asset Class" = "Currency";'
#sql = 'SELECT Ticker FROM etftable WHERE "Niche" = "Intermediate";'
#sql = 'SELECT Ticker FROM etftable WHERE "Focus" = "Silver";'
sql = 'SELECT Ticker FROM etftable WHERE "Category" = "Government Credit";'
#sql = 'SELECT Ticker FROM etftable WHERE "niche"=”Agriculture“;'
 
#create a connection to the database specified above
cnx = db.connect(database)
cur = cnx.cursor()
 
#execute the SQL statement and place the results into a 
#variable called "tickers"
tickers = pd.read_sql(sql, con=cnx)
 
 
#create an empty list
symbList = []
 
#iterate over the DataFrame and append each item into the empty list
for i in range(len(tickers)):
    symbList.append(tickers.iloc[i][0])

symbList = list(set(symbList))
#symbList = ["C", "HOG"] 

#get symbol pairs
symbPairs = list(it.combinations(symbList, 2))

#define empty list    
r = []
ETF_Pair = []
Sharpe_Ratio = []
pvalue_list = []
msg = ""
for i in symbPairs:
    ret = 1
    try:
        (ret,pv,sr) = backtest(i,"yfirst")
        r.append(ret)
        Sharpe_Ratio.append(sr)
        pvalue_list.append(pv)
        print(msg+" --")
    except Exception:
        continue
    ETF_Pair.append(i)

for i in symbPairs:
    ret = 1
    try:
        (ret,pv,sr) = backtest(i,"xfirst")
        r.append(ret)
        Sharpe_Ratio.append(sr)
        pvalue_list.append(pv)
        print(msg+" --")
    except Exception:
        continue
    ETF_Pair.append(i)

#Sharpe ratio list    
print(Sharpe_Ratio)  
#ETF pair list
print(ETF_Pair)
#p_value list
print(pvalue_list)


#best ETF Pair
max_SR = max(Sharpe_Ratio)#maximum Sharpe ratio
max_index = Sharpe_Ratio.index(max_SR) #the position in the list
best_pair = ETF_Pair[max_index]#corresponding ETF pair
print("Best ETF Pair:")
print (best_pair)  
