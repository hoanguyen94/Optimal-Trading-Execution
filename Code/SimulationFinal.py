import numpy as np
from scipy import stats
import pandas as pd
from pathlib import Path
import glob
import ast
import random
import h5py
from tqdm import tqdm


stock = ['CATY']
inputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\NASDAQ_h5'
iniPriceFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Output'
outputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Simulation\\Data'
timeStep = 30    # seconds
noTick = 15
iteration = 3    # define number of simulated observations

""" Estimate parameters for each trading day and average them """

def transform(df):
    """ Transform each cell of the dataframe to numpy array """
    for col in df.columns.values:
        df[col] = df[col].apply(ast.literal_eval)
        df[col] = df[col].apply(np.array)
    return df

def concat(df, limitOrders, cancelOrders):
    """ Append to limit order and cancel order list """
    for col in df.columns:
        for ind in df.index:
            if col == "limitOrders":
                if len(limitOrders) < ind + 1:
                    limitOrders.append(df.loc[ind, col])
                else:
                    limitOrders[ind] = np.concatenate((limitOrders[ind],df.loc[ind, col]))
            elif col == "cancelOrders":
                if len(cancelOrders) < ind + 1:
                    cancelOrders.append(df.loc[ind, col])
                else:
                    cancelOrders[ind] = np.concatenate((cancelOrders[ind],df.loc[ind, col]))
    return None

def delShares(sign, price, vol):
    """ Delete shares executed or canceled """
    global buyDepth, sellDepth, buyQtyMap, sellQtyMap

    if sign == -1:   #update bid Depth
        buyQtyMap[price] -= min(vol, buyQtyMap[price])
        if buyQtyMap[price] == 0:
            del buyQtyMap[price]
            buyDepth.remove(price)

    else:            #update ask Depth
        sellQtyMap[price] -= min(vol, sellQtyMap[price])
        if sellQtyMap[price] == 0:
            del sellQtyMap[price]
            sellDepth.remove(price)
    return None

def updateOShares():
    """ Update no. of outstanding shares present at i ticks from the opposite side """
    global iniBuy, iniSell, Na, Nb

    for dis in np.arange(0, noTick):
        # update no. of outstanding shares in the bid side

        askPrice = round(buyDepth[0] + (dis + 1)/ 100, 2) if len(buyDepth) > 0 else iniSell
        if askPrice in sellDepth:
            Na[dis] = sellQtyMap[askPrice]
        else:
            Na[dis] = 0

        # update no. of outstanding shares in the ask side
        bidPrice = round(sellDepth[0] - (dis + 1)/100, 2) if len(sellDepth) > 0 else iniBuy
        if bidPrice in buyDepth:
            Nb[dis] = buyQtyMap[bidPrice]
        else:
            Nb[dis] = 0
    return Na, Nb

def updateDepth(price, sign):
    """ Insert bid and ask depth """
    i = 0
    if sign == 1:    # ask order
        for p in sellDepth:
            if price < p:
                sellDepth.insert(i, price)
                return None
            elif price == p:
                return None
            i += 1
        sellDepth.insert(i, price)

    else:
        for p in buyDepth:
            if price > p:
                buyDepth.insert(i, price)
                return None
            elif price == p:
                return None
            i += 1
        buyDepth.insert(i, price)
    return None

def appendDf(t):
    """ Append data to save"""
    global tradeBidSize, tradeAskSize, e, m, ev, bestBL, bestAL
    global timeStamp, bestBidPrice, bestAskPrice, bestBidVol, bestAskVol, bidTradeSize, askTradeSize, OFI, marDep, noEvent

    timeStamp.append(t)
    bestBidPrice.append(bestBL[0]) if t != 0 else bestBidPrice.append(buyDepth[0])
    bestAskPrice.append(bestAL[0]) if t != 0 else bestAskPrice.append(sellDepth[0])
    bestBidVol.append(bestBL[1]) if t != 0 else bestBidVol.append(buyQtyMap[buyDepth[0]])
    bestAskVol.append(bestAL[1]) if t != 0 else bestAskVol.append(sellQtyMap[sellDepth[0]])

    # update trade size
    bidTradeSize.append(tradeBidSize)
    askTradeSize.append(tradeAskSize)
    tradeBidSize = 0
    tradeAskSize = 0

    # update OFI
    OFI.append(e)
    e = 0

    # update market depth
    marDep.append(m)
    m = 0

    # update no. of events of
    noEvent.append(ev)
    ev = 0
    return None




if __name__ == '__main__':
    #  time of each trading day in seconds
    T = int(6.5 * 60 * 60)
    for s in tqdm(stock):
        orderFileFmt = inputFolder + '\\' + s + '\\*Order30s.csv'
        inFileList = glob.glob(orderFileFmt)  # no. of files in the input folder
        trainSize = int(len(inFileList) / 2)

        # initialize price for simulation, elements corresponding with stock list
        iniPriceFmt = iniPriceFolder + '\\' + s + '\\*IniPrice.h5'
        iniPrice = glob.glob(iniPriceFmt)
        iniPrice = h5py.File(iniPrice[0], 'r')
        iniPrice = iniPrice[s][:]

        # initialize parameters for limit and cancel orders
        shape = noTick  # no. of distance from the best opposite side
        alphaTrade = np.zeros((trainSize,))
        alphaLimit, alphaCancel = np.zeros((trainSize, shape)), np.zeros((trainSize, shape))
        locTrade, scaleTrade = np.zeros((trainSize,)), np.zeros((trainSize,))
        locLimit, scaleLimit = np.zeros((trainSize, shape)), np.zeros((trainSize, shape))
        locCancel, scaleCancel = np.zeros((trainSize, shape)), np.zeros((trainSize, shape))

        for i in range(trainSize):
            # initialize list of orders
            limitOrders = []
            cancelOrders = []
            tradeOrders = []

            # read data and save them into lists of orders
            df_temp = pd.read_csv(inFileList[i])  # read csv
            df_temp2 = transform(df_temp.loc[:, ['limitOrders', 'cancelOrders']])  # transform to numpy array
            concat(df_temp2, limitOrders, cancelOrders)  # concatenate files, only for limit orders and cancel orders
            tradeOrders += ast.literal_eval(df_temp.loc[0, 'tradeOrders'])  # concatenate arrays for trade orders
            tradeOrders = np.array(tradeOrders)

            # calculate parameters for trade
            alphaTrade[i] = tradeOrders.size / (2 * T)  # probability of number of trades per second
            tradeData = tradeOrders[(tradeOrders <= np.quantile(tradeOrders, 0.95))]  # remove outliers
            locTrade[i], scaleTrade[i] = stats.norm.fit(np.log(tradeData))  # parameters for volume of trade

            for distance in range(shape):

                # calculate parameters for limit orders at a distance i from the opposite best price
                alphaLimit[i][distance] = limitOrders[distance].size / (2 * T)
                if limitOrders[distance].size > 0:
                    limitData = limitOrders[distance][(limitOrders[distance] <= np.quantile(limitOrders[distance], 0.95))]
                    locLimit[i][distance], scaleLimit[i][distance] = stats.norm.fit(np.log(limitData))
                    # calculate parameters for limit orders at a distance i from the opposite best price
                    alphaCancel[i][distance] = cancelOrders[distance].size / (2 * T * limitOrders[distance].mean())
                else:
                    locLimit[i][distance], scaleLimit[i][distance] = 0, 0
                    alphaCancel[i][distance] = 0

                if cancelOrders[distance].size > 0:
                    cancelData = cancelOrders[distance][(cancelOrders[distance] <= np.quantile(cancelOrders[distance], 0.95))]
                    locCancel[i][distance], scaleCancel[i][distance] = stats.norm.fit(np.log(cancelData))
                else:
                    locCancel[i][distance], scaleCancel[i][distance] = 0, 0

        # average estimated parameters across trading day
        aveAlphaTrade = alphaTrade.mean()
        aveLocTrade = locTrade.mean()
        aveScaleTrade = scaleTrade.mean()
        aveAlphaLimit = alphaLimit.mean(axis=0)
        aveLocLimit = locLimit.mean(axis=0)
        aveScaleLimit = scaleLimit.mean(axis=0)
        aveAlphaCancel = alphaCancel.mean(axis=0)
        aveLocCancel = locCancel.mean(axis=0)
        aveScaleCancel = scaleCancel.mean(axis=0)

        ### SIMULATION ###

        # initialize parameters for simulator
        iniBuy = iniPrice[0][1]
        iniSell = iniPrice[1][1]
        buyDepth, sellDepth = [iniBuy], [iniSell]
        buyQtyMap, sellQtyMap = {iniBuy: 1}, {iniSell: 1}
        Na, Nb = np.zeros((shape,)), np.zeros((shape,))  # number of outstanding shares in the ask side present at i ticks from bid side and ask side
        NaCancel, NbCancel = 0, 0  # weighted sum of shares cancelled at price levels from p - i ticks to p - 1
        A = aveAlphaLimit.sum() * 2  # probability of limit orders
        random.seed(42)

        # initialize parameters for trades, OFI and limit orders
        updateBool = [0]  # list for tracking time to save on dataframe
        step = 1  # iteration indicator
        e = 0  # contribution to OFI
        ev = 0  # event
        m = 0  # market depth and last market depth
        tradeBidSize = 0
        tradeAskSize = 0
        bestBL, bestAL = [0, 0], [0, 0]  # [price, volume] for best limit orders and ask limit orders
        lastBL, lastAL = [0, 0], [0, 0]  # [price, volume] for the last best limit orders and ask limit orders

        # initialize parameters for saving dataframe
        timeStamp = []
        bestBidPrice, bestAskPrice = [], []
        bidTradeSize, askTradeSize = [], []
        bestBidVol, bestAskVol = [], []
        OFI = []
        marDep = []
        noEvent = []  # number of events occur during each interval

        t = 0  # initialize time
        while t <= (T * trainSize * iteration):
            print(t)
            # save data to Dataframe
            updateBool.append(t // timeStep)
            if t == 0 or updateBool[step] != updateBool[step - 1]:
                appendDf(t)

            # update no. of outstanding shares
            Na, Nb = updateOShares()

            # compute cancellation rate
            NaCancel = Na @ aveAlphaCancel
            NbCancel = Nb @ aveAlphaCancel

            # Draw a random event in relative probabilities to choose type of orders
            # event 1 for trade, event 2 for limit orders, event 3 for cancellation at ask side, event 4 for cancellation at bid side
            event = random.choices(population=[1, 2, 3, 4], weights=[2 * aveAlphaTrade, A, NaCancel, NbCancel])[0]

            if event == 1:  # market order
                # draw sign: 1 is ask and -1 is bid
                sign = random.choices(population=[-1, 1])[0]

                # draw volume of market orders
                vol = round(np.exp(stats.norm.rvs(loc=aveLocTrade, scale=aveScaleTrade)))

                # update quantity
                if sign == 1:  # ask order
                    # add ask size
                    tradeAskSize += min(vol, buyQtyMap[buyDepth[0]])
                    delShares(-sign, buyDepth[0], vol)

                else:  # bid order
                    # add bid size
                    tradeBidSize += min(vol, sellQtyMap[sellDepth[0]])
                    delShares(-sign, sellDepth[0], vol)

            elif event == 2:  # limit order
                # draw sign: 1 is ask and -1 is bid
                sign = random.choices(population=[-1, 1])[0]

                # draw distance
                distance = random.choices(population=np.arange(0, noTick), weights=aveAlphaLimit)[0]

                # price of limit order
                if sign == 1:  # ask order
                    price = round(buyDepth[0] + (distance + 1) / 100, 2)
                else:  # bid order
                    price = round(sellDepth[0] - (distance + 1) / 100, 2)

                # draw volume
                vol = round(np.exp(stats.norm.rvs(loc=aveLocLimit[distance], scale=aveScaleLimit[distance])))

                # update quantity
                if sign == 1:  # ask order
                    if price in sellDepth:  # if price exists
                        sellQtyMap[price] += vol
                    else:
                        updateDepth(price, sign)
                        sellQtyMap[price] = vol
                else:  # bid order
                    if price in buyDepth:  # if price exists
                        buyQtyMap[price] += vol
                    else:
                        updateDepth(price, sign)
                        buyQtyMap[price] = vol


            elif event == 3:  # cancelation at ask side
                # draw distance
                distance = random.choices(population=np.arange(0, noTick), weights=Na * aveAlphaCancel)[0]

                price = round(buyDepth[0] + (distance + 1) / 100, 2)

                # draw volume
                vol = round(np.exp(stats.norm.rvs(loc=aveLocCancel[distance], scale=aveScaleCancel[distance])))

                # update quantity
                if price in sellDepth:
                    delShares(1, price, vol)

            elif event == 4:  # cancellation at bid side
                # draw distance
                distance = random.choices(population=np.arange(0, noTick), weights=Nb * aveAlphaCancel)[0]

                price = round(sellDepth[0] - (distance + 1) / 100, 2)

                # draw volume
                vol = round(np.exp(stats.norm.rvs(loc=aveLocCancel[distance], scale=aveScaleCancel[distance])))

                # update quantity
                if price in buyDepth:
                    delShares(-1, price, vol)

            # update best bid and best ask price
            try:
                bestBL = [buyDepth[0], buyQtyMap[buyDepth[0]]]
            except Exception:
                pass

            try:
                bestAL = [sellDepth[0], sellQtyMap[sellDepth[0]]]
            except Exception:
                pass

            # update contribution of the event to the bid and ask queues
            e += (bestBL[0] >= lastBL[0]) * bestBL[1] - (bestBL[0] <= lastBL[0]) * lastBL[1] - (bestAL[0] <= lastAL[0]) * \
                 bestAL[1] + (bestAL[0] >= lastAL[0]) * lastAL[1]

            # update total market depth at the best prices and no. of event
            if (bestBL[1] != lastBL[1]) or (bestAL[1] != lastAL[1]) or (bestBL[0] != lastBL[0]) or (bestAL[0] != lastAL[0]):
                m += bestBL[1] + bestAL[1]
                ev += 1

            # update the last best price and volume and last total market depth
            lastBL = [bestBL[0], bestBL[1]]
            lastAL = [bestAL[0], bestAL[1]]

            step += 1

            # Draw waiting time from exponential distribution with parameter S
            S = 2 * aveAlphaTrade + A + NaCancel + NbCancel
            t += stats.expon.rvs(scale=1 / S)

        # save as a dataframe
        df = pd.DataFrame(list(zip(timeStamp, bestBidPrice, bestAskPrice, OFI, bidTradeSize, askTradeSize, marDep,
                                   noEvent, bestBidVol, bestAskVol)),
                          columns=['timeStamp', 'bestBidPrice', 'bestAskPrice', 'OFI', 'bidTradeSize',
                                   'askTradeSize', 'marDep', 'noEvent', 'bestBidVol', 'bestAskVol'])

        # store main data
        sOutputFolder = outputFolder + "\\" + s
        Path(sOutputFolder).mkdir(parents=True, exist_ok=True)
        outputFName = sOutputFolder + "\\" + s + str(T * trainSize).zfill(2) + ".h5"
        file = open(outputFName, 'w+')
        output = pd.HDFStore(outputFName)
        output['df'] = df
        output.close()