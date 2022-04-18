#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import preprocessing
from DataAnalysis import *
import h5py

plt.ion()

inputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Output'

# uncomment the below line if want to test specific stocks
# stock = ['CATY']

def tabular_q_learning(q_func, q_func2, currentState, actionInd, reward, PI, nextState, v, ALPHA, terminal, c=0):
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
    # comment q_func2 line if dont want to plot convergence
    q_func2[currentState + actionInd + (c, )] = q_func[currentState + actionInd]
    return None


def run_episode(q_func, q_func2, df, dictState, marketState, ALPHA, for_training=False, c=0):
    """ Runs one episode """

    # initial price to calculate rewards
    initialPrice = df['midPrice'].iloc[0]

    # initialize for each episode
    reward = 0
    i = 0
    noStep2 = min(noStep, df.shape[0])
    optimalQ = noStep2 * [None]
    optimalAType = noStep2 * [None]
    optimalA = noStep2 * [None]
    accReward, PI = 0, 0
    nextState = []  # 2 private variables
    vMarket, vLimit = 0, 0
    vTest = V
    # for each time
    if for_training:
        for time in np.arange(noStep2):  # remaining time
            for v in np.arange(lot, V + lot, lot):  # remaining shares > 0
                # update state variable: remaining shares
                vState = v_State(v, dictState)

                # update action space
                actionSpace = np.arange(0, v + lot, lot)

                # update current state
                currentState = df.loc[:, marketState].values[noStep2 - 1 - time]

                # add remaining time and remaining shares to current state
                currentState = tuple(np.append(currentState, [time, vState]))

                if time != 0:
                    # for each action to be taken
                    a = 0
                    for action in actionSpace:
                        for actType in ['limit', 'market']:
                            if actType == 'market':
                                actionInd = (0, a)
                                # update immediate rewards
                                reward = (df['bestBidPrice'].iloc[noStep2 - 1 - time] - initialPrice) * min(action, df[
                                    'bestBidVol'].iloc[noStep2 - 1 - time])
                                reward += (df['bestBidPrice'].iloc[noStep2 - 1 - time] - initialPrice) * max(
                                    action - df['bestBidVol'].iloc[noStep2 - 1 - time], 0)

                                # update price impacts
                                PI = df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[noStep2 - 1 - time] @ np.array(
                                    [1, -action, -action, -action]) if action > 0 else 0

                                # update next state
                                vMarket = v - action
                                vState = v_State(vMarket, dictState)
                                nextState = df.loc[:, marketState].values[noStep2 - time]
                                nextState = tuple(np.append(nextState, [time - 1, vState]))

                                # update q value
                                if vMarket > 0:
                                    tabular_q_learning(q_func, q_func2, currentState, actionInd, reward, PI, nextState,
                                                       vMarket, ALPHA, terminal=False, c=c)
                                else:
                                    tabular_q_learning(q_func, q_func2, currentState, actionInd, reward, PI, nextState,
                                                       vMarket, ALPHA, terminal=True, c=c)

                            else:
                                actionInd = (1, a)
                                # update shares filled
                                sharesFilled = min(action, df['2ndFillRo'].iloc[noStep2 - 1 - time])
                                reward = (df['bestAskPrice'].iloc[noStep2 - 1 - time] - initialPrice) * sharesFilled

                                # limit orders affect price of the next period via remaining shares ordered and not executed
                                PI = df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[noStep2 - 1 - time] @ np.array(
                                    [1, sharesFilled - action, 0, sharesFilled - action]) if action > 0 else 0

                                # update next state
                                vLimit = v - sharesFilled

                                vState = v_State(vLimit, dictState)
                                nextState = df.loc[:, marketState].values[noStep2 - time]
                                nextState = tuple(np.append(nextState, [time - 1, vState]))

                                # update q value
                                if vLimit > 0:
                                    tabular_q_learning(q_func, q_func2, currentState, actionInd, reward, PI, nextState,
                                                       vLimit, ALPHA, terminal=False, c=c)
                                else:
                                    tabular_q_learning(q_func, q_func2, currentState, actionInd, reward, PI, nextState,
                                                       vLimit, ALPHA, terminal=True, c=c)

                        # action tuple
                        a += 1
                elif time == 0:
                    # execute all of the remaining shares with market orders
                    actionInd = (0, int(v / lot))

                    # update immediate rewards
                    reward = (df['bestBidPrice'].iloc[noStep2 - 1 - time] - initialPrice) * min(v, df['bestBidVol'].iloc[noStep2 - 1 - time])
                    reward += (df['bestBidPrice'].iloc[noStep2 - 1 - time] - initialPrice) * max(v - df['bestBidVol'].iloc[noStep2 - 1 - time], 0)

                    # update q value
                    tabular_q_learning(q_func, q_func2, currentState, actionInd, reward, PI, nextState, vMarket, ALPHA,
                                       terminal=True, c=c)



    else:
        # for testing
        PI = 0
        for time in np.arange(noStep2):

            vState = v_State(vTest, dictState)

            # update action space
            actionSpace = np.arange(0, vTest + lot, lot)

            # update current state
            currentState = df.loc[:, marketState].values[time]

            # add remaining time and remaining shares to current state, starting from start of the episode
            currentState = tuple(np.append(currentState, [noStep2 - 1 - time, vState]))

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

            # set current action
            if np.array(q)[~np.isnan(q)].size > 0:   #if any of the state is visited
                currentAct = actionList[np.nanargmax(q)]
            elif time == noStep-1 and vTest > 0 and np.isnan(q).all() == True:   # if all of the states are not visited and the end of episode
                currentAct = (0, vTest)
            else:    # if the state is not visited and not the end of episode
                currentAct = (1, 0)

            optimalAType[time] = currentAct[0]
            optimalA[time] = currentAct[1]

            if currentAct[0] == 0:  # market
                reward = (df['bestBidPrice'].iloc[time] - initialPrice + PI) * min(currentAct[1], df['bestBidVol'].iloc[time])
                reward += (df['bestBidPrice'].iloc[time] - initialPrice + PI) * max(currentAct[1] - df['bestBidVol'].iloc[time], 0)
                PI = df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[time] @ np.array(
                    [1, -currentAct[1], -currentAct[1], -currentAct[1]]) if currentAct[1] > 0 else 0
                vTest -= currentAct[1]

            elif currentAct[0] == 1:  # limit
                sharesFilled = min(currentAct[1], df['2ndFillRo'].iloc[time])
                reward = (df['bestAskPrice'].iloc[time] - initialPrice + PI) * sharesFilled
                PI = df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[noStep2 - 1 - time] @ np.array(
                    [1, sharesFilled - currentAct[1], 0, sharesFilled - currentAct[1]]) if currentAct[1] > 0 else 0
                vTest -= sharesFilled

            accReward += reward * (GAMMA ** time)

        return optimalQ, optimalAType, optimalA, accReward


def removeOutliers(df, var, confidence=0.95):
    """ Remove outliers for var, var is a list of variables """
    for i in var:
        df = df[(df[i] <= df[i].quantile(confidence)) & (df[i] >= df[i].quantile(1 - confidence))]
    return df


# function to map action
def cat(x, dictA):
    return dictA[x]

if __name__ == '__main__':
    for s in stock:

        inputFileFmt = inputFolder + '\\' + s + '\\*BestPrice.csv'
        outputFolder = inputFolder + '\\' + s + "\\Training"
        Path(outputFolder).mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(glob.glob(inputFileFmt)[0])

        # define splitting point between train set and test set
        epiList = df['epi'].unique()
        trainSize = int(epiList.size / 2)

        # calculate time step for each episode
        df['microPrice'] = df['bestBidPrice'] * df['bestAskVol'] / (df['bestBidVol'] + df['bestAskVol']) + df[
            'bestAskPrice'] * df['bestBidVol'] / (df['bestBidVol'] + df['bestAskVol'])
        df['microDiff'] = df['microPrice'] - df['MA']
        df['timeStep'] = (df['second'] // 30) * 30
        df['timeStep'] = (df['timeStep'] - (df['timeStep'] // (T * 60)) * (T * 60)) / 30


        # define action set, combining type of orders and volume
        df = df[df['optimalA1'].notnull()]
        df['Act'] = df['optimalAType1'].apply(int).apply(str) + df['optimalA1'].apply(int).apply(str)

        action = [0, 1]
        vol = np.arange(0, V + lot, lot)
        i = 0
        dictA = {}
        for a in action:
            for v in vol:
                dictA[str(a) + str(v)] = i
                i += 1

        df['Act'] = df['Act'].apply(cat, dictA=dictA)
        df['Act'] = df['Act'].astype('int64')

        # store
        df2 = df.drop(['minDate', 'const', 'OFICoeff', 'TICoeff', 'OICoeff'], axis=1)

        # only keep data points if there are some events happening
        df2 = df2[df2['noEvent'] != 0]
        # calculate the correlation matrix
        corr = df2.loc[:, ['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol', 'tradingImb',
                           'movingDiff', 'microDiff']].corr().round(2)

        # plot the heatmap
        fig, axes = plt.subplots(1)
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=0, ax=axes)
        fig.tight_layout()
        plt.savefig(outputFolder + '\\corr.png')
        plt.close()

        # # plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        sns.histplot(data=df2, x="OFI", ax=axes[0, 0])
        sns.histplot(data=df2, x="tradingVol", ax=axes[0, 1])
        sns.histplot(data=df2, x="aveMD", ax=axes[0, 2])
        sns.histplot(data=df2, x="tradingImb", ax=axes[1, 0])
        sns.histplot(data=df2, x="spread", ax=axes[1, 1])
        sns.histplot(data=df2, x="movingDiff", ax=axes[1, 2])
        fig.tight_layout()
        plt.savefig(outputFolder + '\\histBefore.png')
        plt.close()

        # remove outliers
        var = ['OFI', 'priceChange', 'tradingVol', 'aveMD', 'orderImb']
        df2 = removeOutliers(df2, var)

        # # plot historgram after removing outliers
        fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        sns.histplot(data=df2, x="OFI", ax=axes[0, 0])
        sns.histplot(data=df2, x="tradingVol", ax=axes[0, 1])
        sns.histplot(data=df2, x="aveMD", ax=axes[0, 2])
        sns.histplot(data=df2, x="tradingImb", ax=axes[1, 0])
        sns.histplot(data=df2, x="spread", ax=axes[1, 1])
        sns.histplot(data=df2, x="movingDiff", ax=axes[1, 2])
        fig.tight_layout()
        plt.savefig(outputFolder + '\\hisAfterReOutlier.png')
        plt.close()

        # regress on the training set for estimating market impacts
        X = sm.add_constant(df2.loc[df2['epi'] < epiList[trainSize], ['OFI', 'tradingImb', 'orderImb']])
        regPImpact = sm.OLS(df2.loc[df2['epi'] < epiList[trainSize], 'priceChange'], X).fit().get_robustcov_results()

        # save coefficients onto dataframe
        df.loc[df['epi'] < epiList[trainSize], ['const', 'OFICoeff', 'TICoeff', 'OICoeff']] = regPImpact.params
        print("\nResults for market impact: \n", regPImpact.summary())
        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(regPImpact.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionOnAll.png')
        plt.close()

        ### REGRESSION TO CHOOSE STATE SPACE
        # remove outliers
        var = ['OFI', 'tradingVol', 'spread']
        df2 = removeOutliers(df, var)
        # df2 = df[df['optimalA1'] > 0]

        # standardize variables
        scaler = preprocessing.MinMaxScaler().fit(df2[['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep',
                                                       'tradingVol', 'tradingImb', 'movingDiff', 'bestBidVol']])
        X_scaled = scaler.transform(df2[['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol',
                                         'tradingImb', 'movingDiff', 'bestBidVol']])

        # regress all actions, including limit orders and market orders
        X = sm.add_constant(X_scaled)
        OLS = sm.OLS(df2['Act'], X)
        OLS.exog_names[:] = ['Const', 'OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol',
                             'tradingImb', 'movingDiff', 'bestBidVol']
        OLS_res = OLS.fit().get_robustcov_results()
        print(OLS_res.summary())

        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(OLS_res.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionAllAct.png')
        plt.close()

        # regression for limit orders
        df3 = df2[(df2['optimalAType1'] == 1) & (df2['optimalA1'] != 0)]

        # standardize variables
        scaler = preprocessing.MinMaxScaler().fit(df3[['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep',
                                                       'tradingVol', 'tradingImb', 'movingDiff', 'bestBidVol']])
        X_scaled = scaler.transform(df3[['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol',
                                         'tradingImb', 'movingDiff', 'bestBidVol']])

        # regress
        X = sm.add_constant(X_scaled)
        OLS = sm.OLS(df3['Act'], X)
        OLS.exog_names[:] = ['Const', 'OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol',
                             'tradingImb', 'movingDiff', 'bestBidVol']
        OLS_res = OLS.fit().get_robustcov_results()
        print(OLS_res.summary())

        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(OLS_res.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionLimit.png')
        plt.close()

        # regression for market orders
        df3 = df2[df2['optimalAType1'] == 0]

        # standardize variables
        scaler = preprocessing.MinMaxScaler().fit(df3[['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep',
                                                       'tradingVol', 'tradingImb', 'movingDiff', 'bestBidVol']])
        X_scaled = scaler.transform(df3[['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol',
                                         'tradingImb', 'movingDiff', 'bestBidVol']])

        X = sm.add_constant(X_scaled)
        OLS = sm.OLS(df3['Act'], X)
        OLS.exog_names[:] = ['Const', 'OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol',
                             'tradingImb', 'movingDiff', 'bestBidVol']
        OLS_res = OLS.fit().get_robustcov_results()
        print(OLS_res.summary())

        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(OLS_res.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionMarket.png')
        plt.close()

        # regression for type of orders
        df3 = df2[df2['optimalA1'] != 0]
        scaler = preprocessing.MinMaxScaler().fit(df3[['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep',
                                                       'tradingVol', 'tradingImb', 'movingDiff', 'bestBidVol']])
        X_scaled = scaler.transform(df3[['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol',
                                         'tradingImb', 'movingDiff', 'bestBidVol']])

        X = sm.add_constant(X_scaled)
        OLS = sm.OLS(df3['optimalAType1'], X)
        OLS.exog_names[:] = ['Const', 'OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol',
                             'tradingImb', 'movingDiff', 'bestBidVol']
        OLS_res = OLS.fit().get_robustcov_results()
        print(OLS_res.summary())

        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(OLS_res.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionType.png')
        plt.close()

        # discretize market state variables which represent states of the world
        # market variables
        rangec = ['tradingVol', 'tradingImb', 'movingDiff', 'spread', 'OFI', 'aveMD', 'orderImb', 'microPrice', 'bestBidVol', 'microDiff']

        var = ['OFI', 'tradingVol', 'aveMD', 'orderImb', 'spread', 'bestBidVol', 'microPrice']
        # df2 = removeOutliers(df, var)
        df2 = df[df['optimalA1'] > 0]

        print(df2[['tradingVol', 'tradingImb', 'movingDiff', 'spread', 'orderImb']].describe())
        print(df2['tradingVol'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        print(df2['tradingImb'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        print(df2['movingDiff'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        print(df2['spread'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        print(df2['OFI'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        print(df2['aveMD'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))


        # bins corresponding with rangec
        # split points of trading volume, trading imbalance, OFI depend on individual stocks
        rangeq = {'tradingVol': [-np.inf, 0, 100, np.inf],
                  'tradingImb': [-np.inf, 0, 100, np.inf],
                  'movingDiff': [-np.inf, df2['movingDiff'].quantile(0.3), df2['movingDiff'].quantile(0.7), np.inf],
                  'spread': [-np.inf, df2['spread'].quantile(0.3), df2['spread'].quantile(0.7), np.inf],
                  'OFI': [-np.inf, df2['OFI'].quantile(0.3), df2['OFI'].quantile(0.7), np.inf],
                  'aveMD': [-np.inf, df2['aveMD'].quantile(0.3), df2['aveMD'].quantile(0.7), np.inf],
                  'orderImb': [-np.inf, df2['orderImb'].quantile(0.3), df2['orderImb'].quantile(0.7), np.inf],
                  'microPrice': [-np.inf, df2['microPrice'].quantile(0.3), df2['microPrice'].quantile(0.7), np.inf],
                  'bestBidVol': [-np.inf, df2['bestBidVol'].quantile(0.3), df2['bestBidVol'].quantile(0.7), np.inf],
                  'microDiff': [-np.inf, df2['microDiff'].quantile(0.3), df2['microDiff'].quantile(0.7), np.inf]}


        # dictionary for market states
        dictState = {}

        # discretize
        for c in rangec:
            df[c + 'Bin'] = pd.cut(df[c], bins=rangeq[c], labels=range(0, len(rangeq[c]) - 1))
            value = pd.cut(df[c], bins=rangeq[c]).unique()
            key = df[c + 'Bin'].unique()
            dictState[c] = {key[i]: value[i] for i in range(len(key))}

        print(dictState)
        # print(df2['tradingVolBin'].value_counts())
        # print(df2['tradingImbBin'].value_counts())
        # print(df2['spreadBin'].value_counts())
        # print(df2['movingDiffBin'].value_counts())
        # print(df2['OFIBin'].value_counts())

        # test impact of discretized market states on actions
        # depending on stock, change the variables included in the below regressions, may add 'OFIBin', 'microPriceBin' or 'bestBidVolBin'
        # for all actions
        df2 = df
        X = sm.add_constant(
            df2[['tradingVolBin', 'movingDiffBin', 'spreadBin', 'orderImbBin', 'tradingImbBin', 'timeStep']])
        OLS_mod = sm.OLS(df2['Act'], X)
        OLS_res = OLS_mod.fit().get_robustcov_results()

        print(OLS_res.summary())

        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(OLS_res.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionAllActDiscret.png')
        plt.close()

        # for limit orders
        df3 = df2[(df2['optimalAType1'] == 1) & (df2['optimalA1'] != 0)]
        X = sm.add_constant(
            df3[['tradingVolBin', 'movingDiffBin', 'spreadBin', 'orderImbBin', 'tradingImbBin', 'timeStep']])
        OLS_mod = sm.OLS(df3['Act'], X)
        OLS_res = OLS_mod.fit().get_robustcov_results()

        print(OLS_res.summary())

        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(OLS_res.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionLimitDiscret.png')
        plt.close()

        # for market orders
        df3 = df2[(df2['optimalAType1'] == 0) & (df2['optimalA1'] != 0)]
        X = sm.add_constant(
            df3[['tradingVolBin','movingDiffBin', 'spreadBin', 'orderImbBin', 'tradingImbBin', 'timeStep']])
        OLS_mod = sm.OLS(df3['Act'], X)
        OLS_res = OLS_mod.fit().get_robustcov_results()

        print(OLS_res.summary())

        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(OLS_res.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionMarketDiscret.png')
        plt.close()

        # for type of orders
        df2 = df2[df2['optimalA1'] != 0]
        X = sm.add_constant(
            df2[['tradingVolBin', 'movingDiffBin', 'spreadBin', 'orderImbBin', 'tradingImbBin', 'timeStep']])
        OLS_mod = sm.OLS(df2['optimalAType1'], X)
        OLS_res = OLS_mod.fit().get_robustcov_results()

        print(OLS_res.summary())

        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(OLS_res.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionTypeDiscret.png')
        plt.close()

        # divide no. of shares into 5 intervals, for state variable
        rangeV = np.linspace(-lot, V, noLot + 1)
        rangeV_value = pd.cut(rangeV, noLot + 1)
        rangeV_key = pd.cut(rangeV, noLot + 1, labels=range(noLot + 1))
        dictState['vBin'] = {rangeV_key[i]: rangeV_value[i] for i in range(len(rangeV_key))}
        print(dictState)

        del OLS_mod, OLS_res, X, X_scaled, axes, corr, df2, df3, dictA, fig, key, rangeV, rangeV_key, rangeV_value,
        regPImpact, scaler, value, OLS

        #### TRAINING
        # define market state variables, may add 'OFIBin', 'microPriceBin' or 'bestBidVolBin', or change depending on stock
        marketState = ['tradingVolBin', 'tradingImbBin', 'movingDiffBin', 'spreadBin', 'orderImbBin']

        # initialize q function
        # list of dimension of state, excluding time step
        rangeS = []
        for m in marketState:
            n = m[:(len(m) - 3)]
            rangeS.append(len(dictState[n]))

        actionSpace = np.arange(0, V + lot, lot)

        # For DETERMINISTIC EXPERIENCE REPLAY: leave q_func and q_func2 here and in line 566, replace with "for a in [0.3, 0.1]"
        # where 0.3 is the learning rate in the first round and 0.1 is the learning rate in the second round

        # For stock CATY, if run the algo 2 times with learning rate 0.1 and 0.3 SEPERATELY (model 1 and 3) and save in 1 file
        # then put line 563 to line 568 below line 572: c = 0, and in line 566, replace with "for a in [0.3, 0.1]"

        # For other stocks, as only run the algo with learning rate 0.1 only, leave it as is.

        # q_func has dimension of state space x 2 type of actions x action space
        q_func = np.full(rangeS + [int(noStep), noLot + 1, 2, actionSpace.shape[0]], np.nan)

        # for convergence plot, q_func has dimension of state space x 2 type of actions x action space x trainSize
        # set q_func2 = 0 if dont want to plot convergence and comment q_func2 line in tabular q-learning
        q_func2 = np.full(rangeS + [int(noStep), noLot + 1, 2, actionSpace.shape[0], trainSize], np.nan)

        # for each learning rate
        for a in [0.1]:
            c = 0

            df['optimalQ' + str(a)] = None
            df['optimalAType' + str(a)] = None
            df['optimalA' + str(a)] = None
            df['accReward' + str(a)] = None

            for i in tqdm(epiList[:trainSize]):
                if df[df['epi'] == i].shape[0] > 1:
                    run_episode(q_func, q_func2, df[df['epi'] == i], dictState, marketState, a, for_training=True, c=c)
                    c += 1

            for i in tqdm(epiList[trainSize:]):
                if df[df['epi'] == i].shape[0] > 1:
                    optimalQ, optimalAType, optimalA, accReward = run_episode(q_func, q_func2, df[df['epi'] == i],
                                                                              dictState, marketState, a, for_training=False)
                    df.loc[df['epi'] == i, 'optimalQ' + str(a)] = optimalQ
                    df.loc[df['epi'] == i, 'optimalAType' + str(a)] = optimalAType
                    df.loc[df['epi'] == i, 'optimalA' + str(a)] = optimalA
                    df.loc[df['epi'] == i, 'accReward' + str(a)] = accReward

            # store as csv files
            outputFileh5 = outputFolder + "\\" + "BestPrice" + str(a) + str(BETA) + ".h5"
            h5f = h5py.File(outputFileh5, 'w')
            h5f.create_dataset(s, data=q_func2)
            h5f.close()

        # store as csv files
        outputFile = outputFolder + "\\" + str(BETA) + "BestPrice" + ".csv"
        df.to_csv(path_or_buf=outputFile, index=False)




