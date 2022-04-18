"""
Build a book from MD2/MD3 files
"""

import os
import glob
from datetime import datetime, time, timedelta
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path
from ParserMD3 import ParserMD3, MDTickL3

POS_INF = 0x7FFFFFFFFFFFFFFF
NEG_INF = -9223372036854775808

inputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\NASDAQ_md3'
outputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\NASDAQ_h5'
stocks = ['CATY']   #['ANIP', 'ACGL', 'HSIC', 'BRKS', 'CORE', 'CBSH', 'CRUS', 'JCOM', 'CVLT', 'BJRI', 'ALLK', 'ALRM', 'AFYA', 'BOKF']

class Clock(object):
    """ A clock that ticks """
    def _init_(self):
        self.timed_callback_ = SortedDict()
        # Default to epoch
        self.current_time_ = datetime.datetime.fromtimestamp(0)

    def now(self):
        return self.current_time_

    def insert_timed_callback(self, t, cb):
        if t not in self.timed_callback_:
            self.timed_callback_[t] = []
        self.timed_callback_[t].append(cb)

    def advance(self):
        if not self.timed_callback_:
            return False
        for t, all_cb in self.timed_callback_.items():
            self.current_time_ = t
            for cb in all_cb:
                cb()
            self.timed_callback_.pop(t, None)
            break
        return True

class MDBookHandler(object):
    """ Converts a standard MD2/MD3 file into a book """
    def __init__(self):
        self.time = None
        self.bidQty = None
        self.bidPx = None
        self.askPx = None
        self.askQty = None
        self.tradePx = None
        self.tradeQty = None
        self.tradeSide = None
        self.xchTime = None
        self.eqbPx = None
        self.status = "CLOSE"
        self.buyDepth = []  # list of PriceLevel
        self.sellDepth = []  # list of PriceLevel
        self.buyOrderMap = {}  # dict of {orderNum: price}
        self.sellOrderMap = {}  # dict of {orderNum: price}
        self.eventId = 0
        self.lastTick = None

    def onTick(self, tick):
        self.tradePx = None
        self.tradeQty = None
        self.tradeSide = None
        self.lastTick = tick
        self.process(tick)

    def process(self, tick):
        """ Process next tick """
        self.time = tick.timestamp
        self.eventId = tick.eventId

        if tick.type == "STATUS":
            self.status = tick.status

        elif tick.type == "TRADE":
            self.tradePx = tick.price
            self.tradeQty = tick.qty
            self.tradeSide = tick.side
            self.xchTime = tick.xchTime

        elif tick.type == "EQB":
            self.eqbPx = tick.eqbPx

        elif tick.type == "DEPTH":
            # replace mkt None prices with max/min prices
            if tick.price is None:
                tick.price = POS_INF if tick.side == 'BUY' else NEG_INF

            self.updateDepth(tick)
            self.xchTime = tick.xchTime

        elif tick.type == "ORDER":
            # replace mkt None prices with max/min prices
            if tick.command != "DELETE" and tick.price is None:
                tick.price = POS_INF if tick.side == 'BUY' else NEG_INF

            self.updateOrder(tick)
            self.xchTime = tick.xchTime

    def updateDepth(self, tick):
        """ Update depth based on DEPTH tick """
        depth = self.buyDepth if tick.side == "BUY" else self.sellDepth

        idx, found = self.findPrice(depth, tick.side, tick.price)

        if found:
            # update the price level qty
            pxLevel = depth[idx]
            pxLevel.qty = tick.qty

            # delete price level if cleared
            if pxLevel.qty <= 0:
                del depth[idx]

        elif tick.qty > 0:
            # add new price level
            depth.insert(idx, PriceLevel(tick.side, tick.price, tick.qty))
            pxLevel = depth[idx]

        # update bid/ask
        if tick.type == 'DEPTH' and len(depth):
            self.updateBBO(depth, tick.side, tick.price)

    def updateOrder(self, tick):
        """ Update market depth based on ORDER tick """
        depth = self.buyDepth if tick.side == "BUY" else self.sellDepth
        orderMap = self.buyOrderMap if tick.side == "BUY" else self.sellOrderMap
        # if tick.orderNum == '62179824':
        #     print('Stop')
        if tick.command == "ADD":
            self.updateOrder_add(orderMap, depth, tick.side, tick.orderNum,
                                 tick.price, tick.qty)
        elif tick.command == "MODIFY":
            self.updateOrder_mod(orderMap, depth, tick.side, tick.orderNum,
                                 tick.price, tick.qty)
        elif tick.command == "DELETE":
            self.updateOrder_del(orderMap, depth, tick.side, tick.orderNum)

    def updateOrder_add(self, orderMap, depth, side, orderNum, price, qty):
        # get existing price level or add new one
        pxLevel = None
        pxIdx, found = self.findPrice(depth, side, price)
        if found:
            pxLevel = depth[pxIdx]
        else:
            depth.insert(pxIdx, PriceLevel(side, price, 0))
            pxLevel = depth[pxIdx]

        # update pxLevel and orderMap
        pxLevel.addOrder(Order(side, orderNum, price, qty))
        orderMap[orderNum] = price

        # update bid/ask
        self.updateBBO(depth, side, price)

    def updateOrder_mod(self, orderMap, depth, side, orderNum, price, qty):
        # find the order
        if orderNum not in orderMap:
            print(
                "Warning: Cannot process MODIFY op as order {} cannot be found!"
                .format(orderNum))
            return
        oldPrice = orderMap[orderNum]

        # find the price level
        pxIdx, found = self.findPrice(depth, side, oldPrice)
        if not found:
            print(
                "Warning: Cannot process MODIFY op as price {} for order {} cannot be found!"
                .format(price, orderNum))
            return
        pxLevel = depth[pxIdx]

        if oldPrice != price:
            # if the price changed, process as del/add operation
            self.updateOrder_del(orderMap, depth, side, orderNum)
            self.updateOrder_add(orderMap, depth, side, orderNum, price, qty)
        else:
            # modify the order qty only
            pxLevel.modOrder(orderNum, qty)

            # update bid/ask
            self.updateBBO(depth, side, price)

    def updateOrder_del(self, orderMap, depth, side, orderNum):
        # find the order
        if orderNum not in orderMap:
            print(
                "Warning: Cannot process DELETE op as order {} cannot be found!"
                .format(orderNum))
            return
        price = orderMap[orderNum]

        # find the price level
        pxIdx, found = self.findPrice(depth, side, price)
        if not found:
            print(
                "Warning: Cannot process DELETE op as price {} for order {} cannot be found!"
                .format(price, orderNum))
            return
        pxLevel = depth[pxIdx]

        # delete price level if cleared
        pxLevel.delOrder(orderNum)

        if pxLevel.qty <= 0:
            del depth[pxIdx]

        # update bid/ask
        self.updateBBO(depth, pxLevel.side, pxLevel.price)

    def updateBBO(self, depth, side, price):
        if not depth:
            return

        if side == 'BUY':
            # update bid if tick price is the first element
            bidIdx = 0 if depth[0].price != POS_INF else 1
            if len(depth) > bidIdx and depth[bidIdx].price <= price:
                self.bidPx = depth[bidIdx].price
                self.bidQty = depth[bidIdx].qty

                # update ask if new bid has replaced the previous ask
                if len(self.sellDepth) and self.askPx and self.bidPx >= self.askPx and \
                        self.sellDepth[0].price != NEG_INF:
                    self.askPx = self.sellDepth[0].price
                    self.askQty = self.sellDepth[0].qty

        elif side == 'SELL':
            # update ask if tick price is the first element
            askIdx = 0 if depth[0].price != NEG_INF else 1
            if len(depth) > askIdx and depth[askIdx].price >= price:
                self.askPx = depth[askIdx].price
                self.askQty = depth[askIdx].qty

                # update bid if new ask has replaced the previous bid
                if len(self.buyDepth) and self.bidPx and self.askPx <= self.bidPx and \
                        self.buyDepth[0].price != POS_INF:
                    self.bidPx = self.buyDepth[0].price
                    self.bidQty = self.buyDepth[0].qty

    def findPrice(self, depth, side, price):
        """ Returns the index corresponding to price in depth. Buy/sell price
        levels are sorted in descending/ascending order """
        if side == "BUY":
            for i, pq in enumerate(depth):
                if pq.price == price:
                    # found price. return index pointing to element
                    return i, True
                elif pq.price < price:
                    # found 1st element smaller than price
                    return i, False
            # price is smaller than all elements in buy depth
            return len(depth), False
        else:
            for i, pq in enumerate(depth):
                if pq.price == price:
                    # found price. return index pointing to element
                    return i, True
                elif pq.price > price:
                    # found 1st element bigger than price
                    return i, False
            # price is bigger than all elements in sell depth
            return len(depth), False


class PriceLevel(object):
    def __init__(self, side, price, qty):
        self.side = side
        self.price = price
        self.qty = qty
        self.orders = []  # list of Orders

    def addOrder(self, order):
        self.qty += order.qty
        self.orders.append(order)

    def modOrder(self, orderNum, qty):
        idx = self.findOrder(orderNum)
        if idx is None:
            print('ModError: Cannot find order {} {} {}!'.format(
                orderNum, self.side, self.price))
            return
        oldQty = self.orders[idx].qty
        qtyDiff = qty - oldQty
        self.orders[idx].qty = qty
        self.qty += qtyDiff

    def delOrder(self, orderNum):
        idx = self.findOrder(orderNum)
        if idx is None:
            print('DelError: Cannot find order {} {} {}!'.format(
                orderNum, self.side, self.price))
            return
        self.qty -= self.orders[idx].qty
        del self.orders[idx]

    def findOrder(self, orderNum):
        """ Returns the index of the order corresponding to orderNum. Returns
        None if the order cannot be found """
        for i, order in enumerate(self.orders):
            if order.orderNum == orderNum:
                return i
        return None


class Order(object):
    def __init__(self, side, orderNum, price, qty):
        self.side = side
        self.orderNum = orderNum
        self.price = price
        self.qty = qty

def findVol(book, tick):
    """ Find the outstanding shares of tick before cancellation """
    if tick.side == "BUY":
        for i in book.buyDepth:
            if tick.price > i.price:
                return -tick.qtyDiff
            elif tick.price == i.price:
                return i.qty - tick.qtyDiff
    else:
        for i in book.sellDepth:
            if tick.price < i.price:
                return -tick.qtyDiff
            elif tick.price == i.price:
                return i.qty - tick.qtyDiff

def outputFile(inputFile, type, tail='h5'):
    inFileName = inputFile.split('\\')[-1]
    symbol = inFileName.split('_')[-1].split('.')[0]
    dateStr = inFileName.split('_')[0]

    symOutputFolder = outputFolder + "\\" + symbol
    Path(symOutputFolder).mkdir(parents=True, exist_ok=True)
    outFilePath = symOutputFolder + "\\" + dateStr + "_" + symbol + "_" + type + str(interval) + "s." + tail
    return outFilePath, symbol

def parse(inputFile):
    parser = ParserMD3(inputFile)
    book = MDBookHandler()
    i = 1
    e = 0                              # contribution to OFI
    eTrade = 0                         # event for trade
    ev = 0                             # event
    m = 0                              # market depth and last market depth
    updateBool = [0]
    tradeBidSize = 0
    tradeAskSize = 0
    bestBL, bestAL = [0,0], [0,0]      # [price, volume] for best limit orders and ask limit orders
    lastBL, lastAL = [0,0], [0,0]
    best2BL, best2AL = 0, 0            # 2nd best bid and 2nd best ask price
    # [price, volume] for the last best limit orders and ask limit orders
    priceDiff = 0

    for tick in parser:
        book.onTick(tick)
        # if (book.time >= datetime(book.time.year, book.time.month, 1, 15, 59, 30,0)) and book.time.day == 1:
        #     print("stop")

        # if during trading hours
        if book.time.time() >= time(hour=startTradingHour[0], minute=startTradingHour[1]) and book.time.time() <= time(hour=endTradingHour[0],minute=endTradingHour[1],microsecond=1):

            # calculate time difference between each tick and starting trading hour
            timeDiff = book.time - datetime(book.time.year, book.time.month, book.time.day, startTradingHour[0], startTradingHour[1], 0, 0)
            updateBool.append(timeDiff.total_seconds() // interval)

            # append the previous data every 30 seconds if there is no event occurring
            if updateBool[i] != updateBool[i-1] and updateBool[i] != (updateBool[i-1] + 1):
                for j in range(int(updateBool[i] - updateBool[i-1] - 1)):
                    second.append(second[-1] + interval)
                    timeStamp.append(timeStamp[-1] + timedelta(seconds=interval))
                    bestBidPrice.append(bestBidPrice[-1])
                    bestAskPrice.append(bestAskPrice[-1])
                    best2BidPrice.append(best2BidPrice[-1])
                    best2AskPrice.append(best2AskPrice[-1])
                    bestBidVol.append(bestBidVol[-1])
                    bestAskVol.append(bestAskVol[-1])
                    bidTradeSize.append(0)
                    askTradeSize.append(0)
                    OFI.append(0)
                    marDep.append(0)
                    noEvent.append(0)
                    noTrade.append(0)

            # every interval, update:
            if i == 1 or updateBool[i] != updateBool[i-1]:
                second.append(timeDiff.total_seconds())
                timeStamp.append(book.time)
                if i != 1:
                    bestBidPrice.append(bestBL[0])
                    bestAskPrice.append(bestAL[0])
                    best2BidPrice.append(best2BL)
                    best2AskPrice.append(best2AL)
                    bestBidVol.append(bestBL[1])
                    bestAskVol.append(bestAL[1])
                else:
                    bestBidPrice.append(book.buyDepth[0].price)
                    bestAskPrice.append(book.sellDepth[0].price)
                    best2BidPrice.append(book.buyDepth[1].price)
                    best2AskPrice.append(book.sellDepth[1].price)
                    bestBidVol.append(book.buyDepth[0].qty)
                    bestAskVol.append(book.sellDepth[0].qty)

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

                # update no. of trade events
                noTrade.append(eTrade)
                eTrade = 0

            i += 1

            ## starting a new interval
            # update OFI
            try:
                bestBL = [book.buyDepth[0].price, book.buyDepth[0].qty]
                best2BL = book.buyDepth[1].price
            except Exception:
                pass

            try:
                bestAL = [book.sellDepth[0].price, book.sellDepth[0].qty]
                best2AL = book.sellDepth[1].price
            except Exception:
                pass

            # update contribution of the event to the bid and ask queues
            e += (bestBL[0] >= lastBL[0]) * bestBL[1] - (bestBL[0] <= lastBL[0]) * lastBL[1] - (bestAL[0] <= lastAL[0]) * bestAL[1] + (bestAL[0] >= lastAL[0]) * lastAL[1]

            # update total market depth at the best prices and no. of event
            if (bestBL[1] != lastBL[1]) or (bestAL[1] != lastAL[1]) or (bestBL[0] != lastBL[0]) or (bestAL[0] != lastAL[0]):
                m += bestBL[1] + bestAL[1]
                ev += 1


            # if market orders are executed, accumulate quantity within the interval
            if (book.tradeSide == "BUY"):
                tradeBidSize += book.tradeQty
                eTrade += 1
                tradeOrders[0.01].append(book.tradeQty)

            elif (book.tradeSide == "SELL"):
                tradeAskSize += book.tradeQty
                eTrade += 1
                tradeOrders[0.01].append(book.tradeQty)

            # compare the price of new order with the best opposite side
            if tick.side == "BUY":
                priceDiff = round(np.abs(tick.price - bestAL[0]), 2)
                if tick.chgFlag == 1:
                    sign[0.01].append(1)
            elif tick.side == "SELL":
                priceDiff = round((tick.price - bestBL[0]), 2)
                if tick.chgFlag == 1:
                    sign[0.01].append(0)

            # add to the order list orders at a distance k from the best opposite side
            for k in np.linspace(0.01, 0.15, 15):
                if priceDiff == round(k, 2) and (tick.side == "BUY" or tick.side == "SELL"):
                    if tick.chgFlag == 3:     # for cancel orders
                        cancelOrders[round(k, 2)].append(-tick.qtyDiff)

                    elif tick.chgFlag == 1:   # for limit orders
                        limitOrders[round(k, 2)].append(tick.qtyDiff)
                    break

            # update the last best price and volume and last total market depth
            lastBL = [bestBL[0], bestBL[1]]
            lastAL = [bestAL[0], bestAL[1]]



    # save as a dataframe
    df = pd.DataFrame(list(zip(timeStamp, second, bestBidPrice, bestAskPrice, best2BidPrice, best2AskPrice, OFI,
                               bidTradeSize, askTradeSize, marDep, noEvent, bestBidVol, bestAskVol, noTrade)),
                      columns=['timeStamp', 'second', 'bestBidPrice', 'bestAskPrice', 'best2BidPrice', 'best2AskPrice',
                               'OFI',  'bidTradeSize', 'askTradeSize', 'marDep', 'noEvent', 'bestBidVol', 'bestAskVol',
                               'noTrade'])

    df2 = pd.DataFrame({'limitOrders': limitOrders, 'cancelOrders': cancelOrders, 'tradeOrders': tradeOrders, 'sign': sign})


    # store as H5 files
    # store main data
    outputFName, symbol = outputFile(inputFile, 'main', 'h5')
    file = open(outputFName, 'w+')
    output = pd.HDFStore(outputFName)
    output['df'] = df
    output.close()

    # store limit and cancel orders
    outputOName, symbol = outputFile(inputFile, 'Order', 'csv')
    df2.to_csv(path_or_buf=outputOName, index=False)


if __name__ == '__main__':

    # initialize variables
    timeStamp = []
    second = []
    bestBidPrice, bestAskPrice = [], []
    best2BidPrice, best2AskPrice = [], []
    bidTradeSize, askTradeSize = [], []
    bestBidVol, bestAskVol = [], []
    OFI = []
    marDep = []
    noEvent = []  # number of events occur during each interval
    noTrade = []
    startTradingHour = [9, 30]
    endTradingHour = [16, 00]
    interval = 30  # seconds, save data every interval

    limitOrders = {}
    cancelOrders = {}
    sign = {}
    tradeOrders = {}

    # for tick 1 to tick 15, initilize an empty list
    for k in np.linspace(0.01, 0.15, 15):
        limitOrders[round(k, 2)] = []
        cancelOrders[round(k, 2)] = []

    # trade has only 1 list
    tradeOrders[0.01] = []
    sign[0.01] = []   #buy or sell

    for s in stocks:
        inputFileFmt = inputFolder + "\\" + s + "\\*.md3"
        inFileList = glob.glob(inputFileFmt)
        Parallel(n_jobs=30)(delayed(parse)(inputFile) for inputFile in inFileList)
