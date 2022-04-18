"""
MD3 Tick Data Parser
"""

import csv
from datetime import datetime as dt


def dtStr(ts):
    """ Return string representation of datetime """
    return ts.strftime('%d-%m-%Y %H:%M:%S.%f')


def parseToken(token, type):
    return type(token) if token != "None" else None


class ParserMD3(object):
    """ MD3 Parser """
    def __init__(self, filename):
        """ Constructor """
        self.file = open(filename, 'r')

        # skip the headers
        for i in range(4):
            self.file.readline()
        # toggle between new/old fmt variant
        self.newOrderFmt = self.file.readline().split(
            ',')[-1].strip() == "seqNum"

        # initialize csvreader
        self.csvreader = csv.reader(self.file,
                                    delimiter=',',
                                    skipinitialspace=True)
        self.eventId = 0

    def __iter__(self):
        return self

    def __next__(self):
        """ Parse next line and return MDTickL2 """
        try:
            tokens = next(self.csvreader)
        except StopIteration:
            self.file.close()
            raise StopIteration

        tick = MDTickL3()
        tick.eventId = self.eventId
        self.eventId += 1
        tick.timestamp = dt.strptime(tokens[0], "%d-%m-%Y %H:%M:%S.%f")

        tick.type = tokens[1]
        if tick.type == "STATUS":
            tick.status = tokens[2]
            if len(tokens) == 4:  # for backward compatibility
                tick.seqNum = parseToken(tokens[3], str)

        elif tick.type == "TRADE":
            if self.newOrderFmt:
                tick.side = parseToken(tokens[2], str)
                tick.price = parseToken(tokens[3], float)
                tick.qty = parseToken(tokens[4], int)
                tick.xchTime = parseToken(tokens[5], str)
                tick.seqNum = parseToken(tokens[6], str)
            else:
                tick.price = parseToken(tokens[2], float)
                tick.qty = parseToken(tokens[3], int)
                tick.xchTime = parseToken(tokens[4], str)

        elif tick.type == "EQB":
            tick.eqbPx = parseToken(tokens[2], float)
            tick.eqbBidQty = parseToken(tokens[3], int)
            tick.eqbAskQty = parseToken(tokens[4], int)
            if self.newOrderFmt:
                tick.seqNum = parseToken(tokens[5], str)

        elif tick.type == "ORDER":
            if self.newOrderFmt:
                tick.command = parseToken(tokens[2], str)
                tick.side = parseToken(tokens[3], str)
                tick.orderNum = parseToken(tokens[4], str)
                tick.priority = parseToken(tokens[5], int)
                tick.price = parseToken(tokens[6], float)
                tick.qty = parseToken(tokens[7], int)
                tick.chgFlag = parseToken(tokens[8], int)
                tick.xchTime = parseToken(tokens[9], str)
                tick.qtyDiff = parseToken(tokens[10], int)
                tick.seqNum = parseToken(tokens[11], str)
                if len(tokens) > 12:
                    tick.custom = tokens[12:]
            else:
                tick.command = parseToken(tokens[2], str)
                tick.side = parseToken(tokens[3], str)
                tick.orderNum = parseToken(tokens[4], str)
                tick.priority = parseToken(tokens[5], int)
                tick.price = parseToken(tokens[6], float)
                tick.qty = parseToken(tokens[7], int)
                tick.chgFlag = parseToken(tokens[8], int)
                tick.xchTime = parseToken(tokens[9], str)
                if len(tokens) > 10:
                    tick.custom = tokens[10:]

        return tick


class MDTickL3(object):
    """ Tick container (L3) """
    def __init__(self):
        self.type = None  # STATUS / TRADE / EQB / ORDER
        self.timestamp = None
        self.status = None
        self.side = None
        self.price = None
        self.qty = None
        self.qtyDiff = None
        self.xchTime = None
        self.eqbPx = None
        self.eqbBidQty = None
        self.eqbAskQty = None
        self.chgFlag = None
        self.command = None
        self.orderNum = None
        self.priority = None
        self.seqNum = None
        self.custom = None
        self.eventId = 0

    def __str__(self):
        if self.type == "STATUS":
            return "{}, {}, {}, {}".format(dtStr(self.timestamp), self.type,
                                           self.status, self.seqNum)
        elif self.type == "TRADE":
            return "{}, {}, {}, {}, {}, {}, {}".format(dtStr(self.timestamp),
                                                       self.type, self.side,
                                                       self.price, self.qty,
                                                       self.xchTime,
                                                       self.seqNum)
        elif self.type == "EQB":
            return "{}, {}, {}, {}, {}, {}".format(dtStr(self.timestamp),
                                                   self.type, self.eqbPx,
                                                   self.eqbBidQty,
                                                   self.eqbAskQty, self.seqNum)
        elif self.type == "ORDER":
            buffer = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                dtStr(self.timestamp), self.type, self.command, self.side,
                self.orderNum, self.priority, self.price, self.qty,
                self.chgFlag, self.xchTime, self.qtyDiff, self.seqNum)
            if self.custom:
                buffer += ', ' + ', '.join(self.custom)
            return buffer
        else:
            return ""