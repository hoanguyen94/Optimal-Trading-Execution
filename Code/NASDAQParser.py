"""
NASDAQ Parser
"""

import os
import glob
import datetime
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

inputPath = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\NASDAQ'
outputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\NASDAQ_md3'
stocks = ['CATY'] #['ANIP', 'ACGL', 'HSIC', 'BRKS', 'CORE', 'CBSH', 'CRUS', 'JCOM', 'CVLT', 'BJRI', 'ALLK', 'ALRM', 'AFYA', 'BOKF']


class NASDAQParser(object):
    """ NASDAQ Parser """
    def __init__(self):
        """ Constructor """
        pass

    def run(self, inFilePath):
        """ Parse """
        inFileName = inFilePath.split('\\')[-1]
        symbol = inFileName.split('_')[-1].split('.')[0]
        dateStr = inFileName.split('_')[0]
        self.timeBase = datetime.datetime.strptime(dateStr, '%Y%m%d')

        inFile = open(inFilePath, 'r')

        symOutputFolder = outputFolder + "\\" + symbol
        Path(symOutputFolder).mkdir(parents=True, exist_ok=True)

        outFilePath = symOutputFolder + "\\" + dateStr + "_" + symbol + ".md3"
        outFile = open(outFilePath, 'w')

        # skip the csv header
        inFile.readline()

        # print headers
        self.printInstrHeader(outFile, symbol)
        self.printL3Header(outFile)

        self.qtyMap = {}  # map of orderNum, qty
        self.pxMap = {}  # map of orderNum, qty
        self.sideMap = {}  # map of orderNum, qty
        self.seqNo = 0
        buffer = inFile.readline()
        while buffer:
            self.printL3Line(outFile, buffer)
            buffer = inFile.readline()
            self.seqNo += 1

        inFile.close()
        outFile.close()
        print("Wrote", outFilePath)

    def printL3Line(self, outFile, buffer):
        """ Format and print the L3 packet """
        rowData = buffer.split(',')
        if len(rowData) < 8:
            return
        exchTime = int(rowData[0])
        time = self.timeBase + datetime.timedelta(milliseconds=exchTime)
        timeStr = time.strftime('%d-%m-%Y %H:%M:%S.%f')
        orderNum = rowData[2]

        # if orderNum == '62179824':
        #     print('Stop')
        rowType = rowData[3]
        side = None
        if rowType == 'B':
            side = 'BUY'
            self.sideMap[orderNum] = side
        elif rowType == 'S':
            side = 'SELL'
            self.sideMap[orderNum] = side
        elif orderNum in self.sideMap:
            side = self.sideMap[orderNum]
        qty = int(rowData[4])
        qtyDiff = qty
        price = float(rowData[5]) / 10000.0
        priority = None
        chgFlag = None
        orderTag = rowData[6]
        chgReason = ''
        orderType = ''
        if rowType in ['B', 'S', 'D', 'C']:
            newQty = qty
            if rowType == 'B' or rowType == 'S':
                cmd = 'ADD'
                chgFlag = 1
                # add to the qtyMap
                self.qtyMap[orderNum] = qty
                self.pxMap[orderNum] = price

            elif rowType == 'D':
                cmd = 'DELETE'
                chgFlag = 3
                qtyDiff = -self.qtyMap[orderNum]
                self.qtyMap[orderNum] = 0
                price = self.pxMap[orderNum]
                newQty = 0

            elif rowType == 'C':
                cmd = 'MODIFY'
                chgFlag = 3
                if not price:
                    price = self.pxMap[orderNum]
                newQty = self.qtyMap[orderNum] - qty
                qtyDiff = -qty
                self.qtyMap[orderNum] = newQty
                self.pxMap[orderNum] = price

            outFile.write(
                '{}, ORDER, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n'.
                format(timeStr, cmd, side, orderNum, priority, price, newQty,
                       chgFlag, exchTime, qtyDiff, self.seqNo, orderTag))
        elif rowType in ['E', 'F']:
            price = self.pxMap[orderNum]
            filledSide = self.sideMap[orderNum]
            qtyDiff = qty
            chgFlag = 4
            if filledSide == "BUY":
                trdSide = "SELL"
            else:
                trdSide = "BUY"
            newQty = qty
            tradedQty = qty
            if rowType == 'E':
                cmd = 'MODIFY'
                newQty = self.qtyMap[orderNum] - qty
                qtyDiff = -qty
                self.qtyMap[orderNum] = newQty
            else:
                cmd = 'DELETE'
                newQty = 0
                qtyDiff = -self.qtyMap[orderNum]
                tradedQty = self.qtyMap[orderNum]
                self.qtyMap[orderNum] = 0

            outFile.write(
                '{}, ORDER, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n'.
                format(timeStr, cmd, side, orderNum, priority, price, newQty,
                       chgFlag, exchTime, qtyDiff, self.seqNo, orderTag))
            outFile.write('{}, TRADE, {}, {}, {}, {}, {}\n'.format(
                timeStr, trdSide, price, tradedQty, exchTime, self.seqNo))

        elif rowType in ['X']:
            outFile.write('{}, TRADE, , {}, {}, {}, {}\n'.format(
                timeStr, price, qty, exchTime, self.seqNo))

    @staticmethod
    def printInstrHeader(outFile, instr):
        """ Print the instrument header """
        outFile.write("{}, S, 0, 0, 0, 2, 0.01/100000\n".format(instr))

    @staticmethod
    def printL3Header(outFile):
        """ Print header lines (L3)"""
        outFile.write("time, STATUS, status\n")
        outFile.write("time, TRADE, side, price, qty, xchTime\n")
        outFile.write("time, EQB, eqbPrice, eqbBidQty, eqbAskQty\n")
        outFile.write(
            "time, ORDER, ADD/DELETE/MODIFY, side, orderNumber, "
            "priority, price, qty, chgFlag, xchTime, orderTag, chgReason, orderType, seqNum\n"
        )



def parse(inputFile):
    parser = NASDAQParser()
    parser.run(inputFile)


if __name__ == '__main__':
    for s in stocks:
        inputFileFmt = inputPath + "\\" + s + "\\*.csv"
        inFileList = glob.glob(inputFileFmt)
        Parallel(n_jobs=30)(delayed(parse)(inputFile) for inputFile in inFileList)
