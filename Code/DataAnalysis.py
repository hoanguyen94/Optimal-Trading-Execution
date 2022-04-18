#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import glob
import math
from tqdm import tqdm
from pathlib import Path
import h5py


plt.ion()
# Parameters

# In[2]:


V = 500       #shares
lot = 50      #shares per lot
noLot = int(V/lot)   #no of lots
T = 5           #mins to execute T shares

interval = 30   #mins, coefficients of price impacts change every 30 mins
timeStep = 30   #seconds
noStep = int(T * 60/timeStep)     #number of time step for each episode
noEpi = 6.5/(interval/60)  #per day, each day there are 6.5 trading hours

# risk aversion parameter, BETA -> 1: risk-averse property, BETA -> -1: risk-loving property
BETA = 0         # risk aversion para
GAMMA = 1         # discount rate
MA_BW = 10    # bandwidth for MA
# ALPHA = 0.1    # learning rate for training

stock = ['CATY']   #['ANIP', 'ACGL', 'HSIC', 'BRKS', 'CORE', 'CBSH', 'CRUS', 'JCOM', 'CVLT', 'BJRI', 'ALLK', 'ALRM', 'AFYA', 'BOKF']
outputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Output'
inputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\Data\\NASDAQ_h5'


def midPrice(df):
    """ calculate mid price, spread, price change... for each trading day """

    df['midPrice'] = (df['bestBidPrice'] + df['bestAskPrice']) / 2
    df['spread'] = (df['bestAskPrice'] - df['bestBidPrice'])
    df['priceChange'] = df['midPrice'] - df['midPrice'].shift(periods=1)
    df['MA'] = df['midPrice'].rolling(MA_BW, min_periods=1).mean()

    # interval for each episode
    df['subSample'] = df['second'] // (interval * 60)  # every half hour
    df.loc[df['subSample'] > (noEpi - 1), 'subSample'] = noEpi - 1  # adjust for the last trading transactions executed at 16:00:00
    df['step'] = df['second'] // timeStep  # every timeStep interval

    return df


def fillOrder(df):
    """ Specify how limit ask orders are filled """

    # create a new column with lagged values of bid trade size
    df['bAL'] = df['bidTradeSize'].shift(periods=-1)

    # no. of limit ask orders filled at best price
    df['1stFill'] = df['bestAskVol'].where(df['bAL'] > df['bestAskVol'], df['bAL'])

    # no of limit ask orders filled at second best price, or no. of limit ask orders added to the quote and filled at best price
    df['2ndFill'] = df['bAL'] - df['1stFill']

    # round the 2nd fill to lots
    df['2ndFillRo'] = (df['2ndFill'] / lot).round(0) * lot

    # drop the redundant column
    df = df.drop('bAL', axis=1)
    return df

def outputFile(outputFolder, symbol, type, tail='h5'):
    """
    :param outputFolder: folder to store output
    :param symbol: symbol that is processed
    :param type: any string
    :param tail: tail of output files
    :return: path of output file
    """
    symOutputFolder = outputFolder + "\\" + symbol
    Path(symOutputFolder).mkdir(parents=True, exist_ok=True)
    outFilePath = symOutputFolder + "\\" + symbol + "_" + type + "." + tail
    return outFilePath


def utility(d):
    """ utility function, BETA is risk-aversion parameter"""
    if d > 0:
        return (1 - BETA) * d
    else:
        return (1 + BETA) * d


def v_State(v, dictState):
    """ return state of remaining shares """
    for j in range(len(dictState['vBin'])):
        if v in dictState['vBin'][j]:
            return j


def tabular_q_learning(q_func, currentState, actionInd, reward, PI, nextState, v, terminal):
    """Update q_func for a given transition """

    if not terminal:

        # update action space
        actionSpace = np.arange(0, v + lot, lot)

        # initialize next_q
        next_q = 2 * actionSpace.size * [None]
        actionList = 2 * actionSpace.size * [None]
        # for each action to be taken
        a = 0
        ai = 0
        for action in actionSpace:
            for actType in ['limit', 'market']:
                actionInd2 = (0, a) if actType == 'market' else (1, a)
                next_q[ai] = q_func[nextState + actionInd2]
                actionList[ai] = action
                ai += 1
            a += 1

        # optimal q value and action of next state
        ind = ~np.isnan(next_q)    # index for next q having non-null values
        next_q = np.array(next_q)[ind]
        actionList = np.array(actionList)[ind]

        # index for optimal action
        ind = np.argmax(next_q + PI * actionList) if next_q.size > 0 else -1   # if q-value of all current states = 0, = -1
        optimal_nextq = next_q[ind] if ind > -1 else 0
        optimal_nextAct = actionList[ind] if ind > -1 else 0
    else:
        optimal_nextq = 0
        optimal_nextAct = 0

    # update q function
    d = reward + GAMMA * (optimal_nextq + optimal_nextAct * PI) - np.nan_to_num(q_func[currentState + actionInd])
    q_func[currentState + actionInd] = np.nan_to_num(q_func[currentState + actionInd]) + ALPHA * utility(d)

    return None


def run_episode(q_func, df, dictState, for_training=False):
    """ Runs one episode """

    # initial price to calculate rewards
    initialPrice = df['midPrice'].iloc[0]

    # initialize for each episode
    reward = 0
    i = 0
    optimalQ = noStep * [None]
    optimalAType = noStep * [None]
    optimalA = noStep * [None]
    accReward, PI = 0, 0
    vMarket, vLimit = 0, 0
    vState = None
    nextState = ()
    vTest = V
    # for each time
    if for_training:
        for time in np.arange(noStep):  # remaining time
            for v in np.arange(lot, V + lot, lot):  # remaining shares > 0
                # update state variable: remaining shares
                vState = v_State(v, dictState)

                # update action space
                actionSpace = np.arange(0, v + lot, lot)

                # update current state
                # currentState = df.loc[:, marketState].values[noStep - 1 - time]

                # add remaining time and remaining shares to current state
                currentState = tuple([time, vState])

                if time != 0:
                    # for each action to be taken
                    a = 0
                    for action in actionSpace:
                        for actType in ['limit', 'market']:
                            if actType == 'market':
                                actionInd = (0, a)
                                # update immediate rewards
                                reward = (df['bestBidPrice'].iloc[noStep - 1 - time] - initialPrice) * min(action, df[
                                    'bestBidVol'].iloc[noStep - 1 - time])
                                reward += (df['bestBidPrice'].iloc[noStep - 1 - time] - initialPrice) * max(
                                    action - df['bestBidVol'].iloc[noStep - 1 - time], 0)

                                # update price impacts
                                PI = df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[noStep - 1 - time] @ np.array(
                                    [1, -action, -action, -action]) if action > 0 else 0

                                # update next state
                                vMarket = v - action
                                vState = v_State(vMarket, dictState)
                                # nextState = df.loc[:, marketState].values[noStep - time]
                                nextState = tuple([time - 1, vState])

                                # update q value
                                if vMarket > 0:
                                    tabular_q_learning(q_func, currentState, actionInd, reward, PI, nextState,
                                                       vMarket, terminal=False)
                                else:
                                    tabular_q_learning(q_func, currentState, actionInd, reward, PI, nextState,
                                                       vMarket, terminal=True)

                            else:
                                actionInd = (1, a)
                                # update shares filled
                                sharesFilled = min(action, df['2ndFillRo'].iloc[noStep - 1 - time])
                                reward = (df['bestAskPrice'].iloc[noStep - 1 - time] - initialPrice) * sharesFilled

                                # limit orders affect price of the next period via remaining shares ordered and not executed
                                PI = df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[noStep - 1 - time] @ np.array(
                                    [1, sharesFilled - action, 0, sharesFilled - action]) if action > 0 else 0

                                # update next state
                                vLimit = v - sharesFilled
                                # if sharesFilled > 0:
                                #     print("Stop")
                                vState = v_State(vLimit, dictState)
                                # nextState = df.loc[:, marketState].values[noStep - time]
                                nextState = tuple([time - 1, vState])

                                # update q value
                                if vLimit > 0:
                                    tabular_q_learning(q_func, currentState, actionInd, reward, PI, nextState, vLimit,
                                                       terminal=False)
                                else:
                                    tabular_q_learning(q_func, currentState, actionInd, reward, PI, nextState, vLimit,
                                                       terminal=True)

                        # action tuple
                        a += 1
                elif time == 0:
                    # execute all of the remaining shares with market orders
                    actionInd = (0, int(v / lot))

                    # update immediate rewards
                    reward = (df['bestBidPrice'].iloc[noStep - 1 - time] - initialPrice) * min(v,
                        df['bestBidVol'].iloc[noStep - 1 - time])
                    reward += (df['bestBidPrice'].iloc[noStep - 1 - time] - initialPrice) * max(
                        v - df['bestBidVol'].iloc[noStep - 1 - time], 0)

                    # update q value
                    tabular_q_learning(q_func, currentState, actionInd, reward, PI, nextState, vMarket, terminal=True)

    else:
        # for testing
        PI = 0
        for time in np.arange(noStep):

            vState = v_State(vTest, dictState)

            # update action space
            actionSpace = np.arange(0, vTest + lot, lot)

            # update current state
            # currentState = df.loc[:, marketState].values[time]

            # add remaining time and remaining shares to current state, starting from start of the episode
            currentState = tuple([noStep - 1 - time, vState])

            # for each action to be taken
            q = 2 * actionSpace.size * [None]
            actionList = 2 * actionSpace.size * [None]
            a = 0
            ai = 0
            for action in actionSpace:
                for actType in ['limit', 'market']:
                    actionInd = (0, a) if actType == 'market' else (1, a)
                    q[ai] = q_func[currentState + actionInd]
                    actionList[ai] = (actionInd[0], action)
                    ai += 1
                a += 1

            # optimal q value and action of next state
            optimalQ[time] = np.nanmax(q) if np.array(q)[~np.isnan(q)].size > 0 else 0
            currentAct = actionList[np.nanargmax(q)] if np.array(q)[~np.isnan(q)].size > 0 else (1, 0)
            optimalAType[time] = currentAct[0]
            optimalA[time] = currentAct[1]

            if currentAct[0] == 0:  # market
                reward = (df['bestBidPrice'].iloc[time] - initialPrice + PI) * min(currentAct[1], df['bestBidVol'].iloc[time])
                reward += (df['bestBidPrice'].iloc[time] - initialPrice + PI) * max(currentAct[1] - df['bestBidVol'].iloc[time], 0)
                PI = df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[time] @ np.array([1, -currentAct[1], -currentAct[1], -currentAct[1]]) if currentAct[1] > 0 else 0
                vTest -= currentAct[1]

            elif currentAct[0] == 1:  # limit
                sharesFilled = min(currentAct[1], df['2ndFillRo'].iloc[time])
                reward = (df['bestAskPrice'].iloc[time] - initialPrice + PI) * sharesFilled
                PI = df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[noStep - 1 - time] @ np.array(
                    [1, sharesFilled - currentAct[1], 0, sharesFilled - currentAct[1]]) if currentAct[1] > 0 else 0
                vTest -= sharesFilled

            accReward += reward * (GAMMA ** time)

        return optimalQ, optimalAType, optimalA, accReward

if __name__ == '__main__':
    for s in stock:
        inputFileFmt = inputFolder + '\\' + s + '\\*main30S.h5'
        symOutputFolder = outputFolder + "\\" + s
        Path(symOutputFolder).mkdir(parents=True, exist_ok=True)

        # load data
        inFileList = glob.glob(inputFileFmt)
        df = pd.DataFrame()
        iniPrice = np.zeros((2, len(inFileList)))

        # preprocess for each file corresponding with each trading day and concatenate them
        i = 0
        for f in inFileList:
            df_temp = pd.read_hdf(f)
            df_temp = midPrice(df_temp)
            df_temp = fillOrder(df_temp)
            iniPrice[0][i] = df_temp.loc[0, 'bestBidPrice']
            iniPrice[1][i] = df_temp.loc[0, 'bestAskPrice']
            frames = [df, df_temp]
            df = pd.concat(frames)
            i += 1

        df.reset_index(inplace=True,drop=True)
        df['minDate'] = np.array(df['timeStamp'], dtype='datetime64[m]')                                        # round date to minutes
        df['epi'] = (df['minDate'] - df.loc[0, 'minDate']).dt.total_seconds() // (T * 60)                       #episodes for trading, lasting T minutes
        df = df.fillna(0)
        df.reset_index(inplace=True,drop=True)

        # create variables
        df['tradingVol'] = df['bidTradeSize'] + df['askTradeSize']
        df['tradingImb'] = df['bidTradeSize'] - df['askTradeSize']
        df['aveMD'] = df['marDep']/ (2 * df['noEvent'])               # average market depth
        df['aveMD'] = df['aveMD'].where(df['aveMD'].notnull(), 0)
        df['orderImb'] = df['bestBidVol'] - df['bestAskVol']
        df['movingDiff'] = df['midPrice'] - df['MA']
        df['TISquared'] = df['tradingImb'] * df['tradingImb'].abs()
        df['TVSquaredRoot'] = df['tradingVol'].apply(np.sqrt)


        # save summary statistics
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.text(0.01, 0.05, str(df[['midPrice', 'priceChange', 'OFI', 'aveMD', 'spread']].describe()),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.tight_layout()
        plt.savefig(symOutputFolder + '\\SummaryStatistics1.png')
        plt.close()

        ax = plt.subplot(111, frame_on=False, autoscale_on=True)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.text(0.01, 0.05, str(df[['tradingVol', 'tradingImb', 'orderImb', 'movingDiff']].describe()),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.tight_layout()
        plt.savefig(symOutputFolder + '\\SummaryStatistics2.png')
        plt.close()

        # save summary statistics for price impact investigation
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.text(0.01, 0.05, str(df[['tradingVol', 'OFI', 'aveMD', 'tradingImb', 'orderImb']].quantile([0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95])),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.tight_layout()
        plt.savefig(symOutputFolder + '\\PriceImpactSummaryStatBefore.png')
        plt.close()

        # plot bid and ask price per 30-second on average
        df_plot = df.loc[:,['step','midPrice', 'priceChange', 'tradingVol', 'OFI','subSample', 'aveMD', 'tradingImb', 'spread', 'orderImb']].groupby(['step']).mean()
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        axs[0, 0].plot(df_plot['midPrice'], label='Mid Price', lw=3)
        axs[0, 0].set_title("Mid-price")
        axs[1, 0].plot(df_plot['priceChange'], label='Mid-price Change', lw=3)
        axs[1, 0].set_title("Mid-price Change")
        axs[1, 0].set_xlabel("30-second", fontsize=18)
        axs[0, 1].plot(df_plot['tradingVol'], label='Trading Volume')
        axs[0, 1].set_title("Trading Volume")
        axs[1, 1].plot(df_plot['OFI'], label='Order Flow Imbalance')
        axs[1, 1].set_title("Order Flow Imbalance")
        axs[1, 1].set_xlabel("30-second", fontsize=18)
        axs[1, 2].plot(df_plot['aveMD'], label='Average Market Depth')
        axs[1, 2].set_title("Average Market Depth")
        axs[0, 2].plot(df_plot['tradingImb'], label='Trading Imbalance')
        axs[0, 2].set_title("Trading Imbalance")
        axs[1, 2].set_xlabel("30-second", fontsize=18)
        fig.tight_layout()
        plt.savefig(symOutputFolder + '\\Per30s.png')
        plt.close()

        # plot bid and ask price per half-hour
        df_plotH = df.loc[:, ['step','midPrice', 'priceChange', 'tradingVol', 'OFI','subSample', 'aveMD', 'tradingImb', 'orderImb']].groupby(['subSample']).mean()
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        axs[0, 0].plot(df_plotH['midPrice'], label='Mid Price', lw=3)
        axs[0, 0].set_title("Mid-price")
        axs[1, 0].plot(df_plotH['priceChange'], label='Mid-price Change', lw=3)
        axs[1, 0].set_title("Mid-price Change")
        axs[1, 0].set_xlabel("Half-hour", fontsize=18)
        axs[0, 1].plot(df_plotH['tradingVol'], label='Trading Volume')
        axs[0, 1].set_title("Trading Volume")
        axs[1, 1].plot(df_plotH['OFI'], label='Order Flow Imbalance')
        axs[1, 1].set_title("Order Flow Imbalance")
        axs[1, 1].set_xlabel("Half-hour", fontsize=18)
        axs[1, 2].plot(df_plotH['aveMD'], label='Average Market Depth')
        axs[1, 2].set_title("Average Market Depth")
        axs[0, 2].plot(df_plotH['tradingImb'], label='Trading Imbalance')
        axs[0, 2].set_title("Trading Imbalance")
        axs[1, 2].set_xlabel("Half-hour", fontsize=18)
        fig.tight_layout()
        plt.savefig(symOutputFolder + '\\PerHalfHour.png')
        plt.close()
        del df_plotH

        # box plot
        selection = ['priceChange', 'tradingVol', 'OFI', 'aveMD', 'tradingImb', 'orderImb']
        title = ['Price Change', 'Trading Volume', 'Order Flow Imbalance', 'Average Market Depth', 'Trading Imbalance', 'Order Imbalance']
        fig, axes = plt.subplots(1, len(selection), figsize=(15, 5))
        for i, col in enumerate(selection):
            ax = sns.boxplot(y=df[col], ax=axes.flatten()[i])
            ax.set_title(title[i])
        fig.tight_layout()
        plt.savefig(symOutputFolder + '\\BoxPlotBeforeOutliers.png')
        plt.close()

        # plot autocorrelation graph
        # dfTrade = df[df['netTrade']!=0]
        fig, ax = plt.subplots(figsize=(15, 5))
        sm.graphics.tsa.plot_acf(df_plot['priceChange'], lags=50, ax=ax, label='Price Change');     #; is to suppress 1 redundant graph
        sm.graphics.tsa.plot_acf(df_plot['OFI'], lags=50, ax=ax, label='OFI at First Level Price');
        sm.graphics.tsa.plot_acf(df_plot['tradingImb'], lags=50, ax=ax, label='Trading Imbalance');
        sm.graphics.tsa.plot_acf(df_plot['orderImb'], lags=50, ax=ax, label='Order Imbalance at First Level Price');
        ax.set_xlabel("Lags", fontsize=18)
        ax.set_ylabel("Autocorrelation", fontsize=18)
        ax.legend(loc="upper right")
        fig.tight_layout()
        plt.savefig(symOutputFolder + '\\ACF.png')
        plt.close()
        del df_plot

        # calculate the correlation matrix
        corr = df.loc[:, ['tradingVol', 'OFI', 'aveMD', 'tradingImb', 'TISquared', 'TVSquaredRoot', 'orderImb']].corr().round(2)

        # plot the heatmap
        fig, ax = plt.subplots(1)
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax=ax)
        fig.tight_layout()
        plt.savefig(symOutputFolder + '\\Corr.png')
        plt.close()

        # run regression with all data
        # remove outliers
        df2 = df[(df['OFI'] <= df['OFI'].quantile(.95)) & (df['OFI'] >= df['OFI'].quantile(.05))]
        df2 = df2[(df2['priceChange'] <= df2['priceChange'].quantile(.95)) & (df2['priceChange'] >= df2['priceChange'].quantile(.05))]
        df2 = df2[(df2['tradingVol'] <= df2['tradingVol'].quantile(.95)) & (df2['tradingVol'] >= df2['tradingVol'].quantile(.05))]
        df2 = df2[(df2['aveMD'] <= df2['aveMD'].quantile(.95)) & (df2['aveMD'] >= df2['aveMD'].quantile(.05))]
        df2 = df2[(df2['orderImb'] <= df2['orderImb'].quantile(.95)) & (df2['orderImb'] >= df2['orderImb'].quantile(.05))]

        # plot histograms after removing outliers
        fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        sns.histplot(data=df2, x="OFI", ax=axes[0,0], bins=30)
        sns.histplot(data=df2, x="tradingVol", ax=axes[0,1], bins=30)
        sns.histplot(data=df2, x="aveMD", ax=axes[1,0], bins=30)
        sns.histplot(data=df2, x="tradingImb", ax=axes[1,1], bins=30)
        sns.histplot(data=df2, x="orderImb", ax=axes[0, 2], bins=30)
        sns.histplot(data=df2, x="priceChange", ax=axes[1, 2], bins=30)
        fig.tight_layout()
        plt.savefig(symOutputFolder + '\\HistogramAfterOutliers.png')
        plt.close()

        # box plot after removing outliers
        fig, axes = plt.subplots(1, len(selection), figsize=(15, 5))
        for i, col in enumerate(selection):
            ax = sns.boxplot(y=df2[col], ax=axes.flatten()[i])
            ax.set_title(title[i])
        fig.tight_layout()
        plt.savefig(symOutputFolder + '\\BoxPlotAfterOutliers.png')
        plt.close()
        del selection, title

        # save summary statistics after removing outliers
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.text(0.01, 0.05, str(df2[['tradingVol', 'OFI', 'aveMD', 'tradingImb', 'orderImb']].describe()),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.tight_layout()
        plt.savefig(symOutputFolder + '\\PriceImpactSummaryStatAfter.png')
        plt.close()

        # scatter plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        sns.scatterplot(data=df2, x="OFI", y="priceChange", ax=axes[0, 0])
        sns.scatterplot(data=df2, x="tradingVol", y="priceChange", ax=axes[0, 1])
        sns.scatterplot(data=df2, x="TVSquaredRoot", y="priceChange", ax=axes[0, 2])
        sns.scatterplot(data=df2, x="aveMD", y="priceChange", ax=axes[1, 0])
        sns.scatterplot(data=df2, x="tradingImb", y="priceChange", ax=axes[1, 1])
        sns.scatterplot(data=df2, x="TISquared", y="priceChange", ax=axes[1, 2])
        fig.tight_layout()
        plt.show()
        plt.savefig(symOutputFolder + '\\VsPriceChg.png')
        plt.close()

        # regress on the whole data
        # specification 1
        X = sm.add_constant(df2.loc[:,['OFI', 'tradingVol', 'aveMD', 'tradingImb', 'noEvent', 'orderImb']])
        regPImpact = sm.OLS(df2['priceChange'], X).fit().get_robustcov_results()
        print("\nResults for all data: \n", regPImpact.summary())
        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(regPImpact.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(symOutputFolder + '\\RegressionOnAll.png')
        plt.close()

        # specification 2
        X = sm.add_constant(df2.loc[:,['OFI', 'TVSquaredRoot', 'aveMD', 'tradingImb', 'noEvent', 'orderImb']])
        regPImpact = sm.OLS(df2['priceChange'], X).fit().get_robustcov_results()
        print("\nResults for all data: \n", regPImpact.summary())
        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(regPImpact.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(symOutputFolder + '\\RegressionOnAll_test1.png')
        plt.close()

        # specification 3
        X = sm.add_constant(df2.loc[:,['OFI', 'tradingVol', 'aveMD', 'TISquared', 'noEvent', 'orderImb']])
        regPImpact = sm.OLS(df2['priceChange'], X).fit().get_robustcov_results()
        print("\nResults for all data: \n", regPImpact.summary())
        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(regPImpact.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(symOutputFolder + '\\RegressionOnAll_test2.png')
        plt.close()

        # specification 4
        X = sm.add_constant(df2.loc[:,['OFI', 'tradingImb', 'orderImb']])
        regPImpact = sm.OLS(df2['priceChange'], X).fit().get_robustcov_results()
        print("\nResults for all data: \n", regPImpact.summary())
        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(regPImpact.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(symOutputFolder + '\\RegressionOnAll_parsimonious.png')
        plt.close()

        # initialize column for predicted price impact
        df[['const', 'OFICoeff', 'TICoeff']] = np.nan

        # regress for each half hour of the day
        halfHour = df['subSample'].unique()
        for i in halfHour:
            df2 = df.loc[df['subSample']==i, ['timeStamp', 'OFI', 'priceChange', 'tradingVol', 'tradingImb', 'aveMD', 'orderImb']]

            # remove outliers
            df2 = df2[(df2['OFI'] <= df2['OFI'].quantile(.95)) & (df2['OFI'] >= df2['OFI'].quantile(.05))]
            df2 = df2[(df2['priceChange'] <= df2['priceChange'].quantile(.95)) & (df2['priceChange'] >= df2['priceChange'].quantile(.05))]

            # plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            sns.scatterplot(data=df2, x="OFI", y="priceChange", ax=axes[0])
            sns.scatterplot(data=df2, x="tradingImb", y="priceChange", ax=axes[1])
            fig.tight_layout()
            plt.show()
            plt.savefig(symOutputFolder + '\\' + str(i) + 'HalfHour.png')
            plt.close()

            # regression
            X = sm.add_constant(df2.loc[:,['OFI', 'tradingImb', 'orderImb']])
            regPImpact2 = sm.OLS(df2['priceChange'], X).fit().get_robustcov_results()

            # calculate predicted price change, even for outliers
            df.loc[df['subSample']==i,['const', 'OFICoeff', 'TICoeff', 'OICoeff']] = regPImpact2.params
            print("\nResults for half-hour: ", i, "\n",  regPImpact2.summary())

            # save as pic
            plt.rc('figure', figsize=(12, 7))
            plt.text(0.01, 0.05, str(regPImpact2.summary()), {'fontsize': 10}, fontproperties='monospace')
            plt.savefig(symOutputFolder + '\\Regression' + str(i) + '.png')
            plt.close()

        # discretize market state variables which represent states of the world
        # market variables
        # rangec = ['OFI', 'tradingImb', 'orderImb']

        # bins corresponding with rangec, no. of time step and total shares V
        rangeq = [noStep, noLot+1]
        j = 0

        # dictionary for market states
        dictState = {}

        # divide no. of shares into 5 intervals, for state variable
        rangeV = np.linspace(-lot, V, rangeq[-1])
        rangeV_value = pd.cut(rangeV, rangeq[-1])
        rangeV_key = pd.cut(rangeV, rangeq[-1], labels=range(rangeq[-1]))
        dictState['vBin'] = {rangeV_key[i]: rangeV_value[i] for i in range(len(rangeV_key))}
        print(dictState)

        # define market state
        marketState = []

        # initialize q function
        # dimension of states
        actionSpace = np.arange(0, V + lot, lot)

        epiList = df['epi'].unique()
        trainSize = int(epiList.size/2)

        for i in tqdm(epiList):
            ALPHA = 1
            a = 1
            if df[df['epi'] == i].shape[0] > 1:
                # q_func has dimension of state space x 2 type of actions x action space
                q_func = np.full(rangeq + [2, actionSpace.shape[0]], np.nan)

                run_episode(q_func, df[df['epi'] == i], dictState, for_training=True)
                optimalQ, optimalAType, optimalA, accReward = run_episode(q_func, df[df['epi'] == i], dictState, for_training=False)
                df.loc[df['epi'] == i, 'optimalQ' + str(a)] = optimalQ
                df.loc[df['epi'] == i, 'optimalAType' + str(a)] = optimalAType
                df.loc[df['epi'] == i, 'optimalA' + str(a)] = optimalA
                df.loc[df['epi'] == i, 'accReward' + str(a)] = accReward

        # store df as csv files
        outputFileDf = outputFile(outputFolder, s, 'BestPrice', tail='csv')
        df.to_csv(path_or_buf=outputFileDf, index=False)

        # store initial price as h5
        iniPriceFile = outputFile(outputFolder, s, 'IniPrice', tail='h5')
        h5f = h5py.File(iniPriceFile, 'w')
        h5f.create_dataset(s, data=iniPrice)
        h5f.close()







