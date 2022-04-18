#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import preprocessing
from Training import *
import h5py

plt.ion()

stock = ['CATY']
inputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Simulation\\Data'
testFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Output'
MA_BW = 10    #bandwidth for MA

if __name__ == '__main__':
    for s in stock:

        inputFileFmt = inputFolder + '\\' + s + '\\*.h5'
        outputFolder = inputFolder + '\\' + s + "\\Training"
        Path(outputFolder).mkdir(parents=True, exist_ok=True)
        inputFile = glob.glob(inputFileFmt)
        df = pd.read_hdf(inputFile[0])
        df = fillOrder(df)

        testFileFmt = testFolder + '\\' + s + '\\Training\\*BestPrice.csv'
        dfTest = pd.read_csv(glob.glob(testFileFmt)[0])
        dfTest = dfTest[dfTest['optimalQ0.1'].notnull()]

        # calculate time step for each episode
        df['epi'] = df['timeStamp'] // (T * 60)
        df['midPrice'] = (df['bestBidPrice'] + df['bestAskPrice'])/2
        df['spread'] = (df['bestAskPrice'] - df['bestBidPrice'])
        df['priceChange'] = df['midPrice'] - df['midPrice'].shift(periods=1)
        df['tradingVol'] = df['bidTradeSize'] + df['askTradeSize']
        df['tradingImb'] = df['bidTradeSize'] - df['askTradeSize']
        df['aveMD'] = df['marDep'] / (2 * df['noEvent'])  # average market depth
        df['aveMD'] = df['aveMD'].where(df['aveMD'].notnull(), 0)

        df['timeStep'] = (df['timeStamp'] // 30) * 30
        df['timeStep'] = (df['timeStep'] - (df['timeStep'] // (T * 60)) * (T * 60)) / 30
        df['microPrice'] = df['bestBidPrice'] * df['bestAskVol'] / (df['bestBidVol'] + df['bestAskVol']) + df[
            'bestAskPrice'] * df['bestBidVol'] / (df['bestBidVol'] + df['bestAskVol'])
        df['orderImb'] = df['bestBidVol'] - df['bestAskVol']
        df['MA'] = df['midPrice'].rolling(MA_BW, min_periods=1).mean()
        df['movingDiff'] = df['midPrice'] - df['MA']

        # save summary statistics
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.text(0.01, 0.05, str(df[['midPrice', 'priceChange', 'OFI', 'aveMD', 'spread']].describe()),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.tight_layout()
        plt.savefig(outputFolder + '\\SummaryStatistics1.png')
        plt.close()

        ax = plt.subplot(111, frame_on=False, autoscale_on=True)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.text(0.01, 0.05, str(df[['tradingVol', 'tradingImb', 'orderImb', 'movingDiff']].describe()),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.tight_layout()
        plt.savefig(outputFolder + '\\SummaryStatistics2.png')
        plt.close()

        # only keep data points if there are some events happening
        df2 = df[df['noEvent'] != 0]
        # calculate the correlation matrix
        corr = df2.loc[:, ['OFI', 'aveMD', 'spread', 'microPrice', 'orderImb', 'timeStep', 'tradingVol', 'tradingImb', 'movingDiff']].corr().round(2)

        # plot the heatmap
        fig, ax = plt.subplots(1)
        sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=0, ax=ax)
        fig.tight_layout()
        plt.savefig(outputFolder + '\\corr.png')
        plt.close()

        # plot
        # fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        # sns.histplot(data=df2, x="OFI", ax=axes[0,0])
        # sns.histplot(data=df2, x="tradingVol", ax=axes[0,1])
        # sns.histplot(data=df2, x="aveMD", ax=axes[0,2])
        # sns.histplot(data=df2, x="tradingImb", ax=axes[1,0])
        # sns.histplot(data=df2, x="spread", ax=axes[1,1])
        # sns.histplot(data=df2, x="movingDiff", ax=axes[1,2])
        # fig.tight_layout()
        # plt.savefig(outputFolder + '\\histBefore.png')
        # plt.close()

        # remove outliers
        var = ['OFI', 'priceChange', 'tradingVol', 'aveMD', 'orderImb']
        df2 = removeOutliers(df2, var)

        # plot historgram after removing outliers
        # fig, axes = plt.subplots(2, 3, figsize=(15, 5))
        # sns.histplot(data=df2, x="OFI", ax=axes[0, 0])
        # sns.histplot(data=df2, x="tradingVol", ax=axes[0, 1])
        # sns.histplot(data=df2, x="aveMD", ax=axes[0, 2])
        # sns.histplot(data=df2, x="tradingImb", ax=axes[1, 0])
        # sns.histplot(data=df2, x="spread", ax=axes[1, 1])
        # sns.histplot(data=df2, x="movingDiff", ax=axes[1, 2])
        # fig.tight_layout()
        # plt.savefig(outputFolder + '\\hisAfterReOutlier.png')
        # plt.close()

        # regress on the training set for estimating market impacts
        X = sm.add_constant(df2[['OFI', 'tradingImb', 'orderImb']])
        regPImpact = sm.OLS(df2['priceChange'], X).fit().get_robustcov_results()

        # save coefficients onto dataframe
        df[['const', 'OFICoeff', 'TICoeff', 'OICoeff']] = regPImpact.params
        print("\nResults for market impact: \n", regPImpact.summary())
        # save as pictures
        plt.rc('figure', figsize=(12, 7))
        plt.text(0.01, 0.05, str(regPImpact.summary()), {'fontsize': 10}, fontproperties='monospace')
        plt.savefig(outputFolder + '\\RegressionOnAll.png')
        plt.close()

        var = ['OFI', 'tradingVol', 'aveMD', 'orderImb', 'spread']
        df2 = removeOutliers(df, var)
        print(df2[['tradingVol', 'tradingImb', 'movingDiff', 'spread', 'orderImb']].describe())
        print(df2[['bestBidVol', 'microPrice', 'OFI', 'aveMD']].describe())
        # print(df2['tradingVol'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        # print(df2['tradingImb'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        # print(df2['movingDiff'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        # print(df2['spread'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        # print(df2['OFI'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        # print(df2['aveMD'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))
        # print(df2['orderImb'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]))

        # discretize market state variables which represent states of the world
        # market variables
        rangec = ['tradingVol', 'tradingImb', 'movingDiff', 'spread', 'OFI', 'aveMD', 'orderImb', 'bestBidVol', 'microPrice']

        # bins corresponding with rangec
        # split points of trading volume, trading imbalance, OFI depend on individual stocks
        rangeq = {'tradingVol': [-np.inf, df2['tradingVol'].quantile(0.3), df2['tradingVol'].quantile(0.7), np.inf],
                  'tradingImb': [-np.inf, df2['tradingImb'].quantile(0.3), df2['tradingImb'].quantile(0.7), np.inf],
                  'movingDiff': [-np.inf, df2['movingDiff'].quantile(0.3), df2['movingDiff'].quantile(0.7), np.inf],
                  'spread': [-np.inf, df2['spread'].quantile(0.3), df2['spread'].quantile(0.7), np.inf],
                  'OFI': [-np.inf, df2['OFI'].quantile(0.3), df2['OFI'].quantile(0.7), np.inf],
                  'aveMD': [-np.inf, df2['aveMD'].quantile(0.3), df2['aveMD'].quantile(0.7), np.inf],
                  'orderImb': [-np.inf, df2['orderImb'].quantile(0.3), df2['orderImb'].quantile(0.7), np.inf],
                  'bestBidVol': [-np.inf, df2['bestBidVol'].quantile(0.3), df2['bestBidVol'].quantile(0.7), np.inf],
                  'microPrice': [-np.inf, df2['microPrice'].quantile(0.3), df2['microPrice'].quantile(0.7), np.inf]}

        # store as csv files
        outputFile = outputFolder + "\\" + "DictState.csv"
        dfState = pd.DataFrame(rangeq)
        dfState.to_csv(path_or_buf=outputFile, index=False)

        # dictionary for market states
        dictState = {}

        # discretize
        for c in rangec:
            df[c+'Bin'] = pd.cut(df[c], bins=rangeq[c], labels=range(0, len(rangeq[c])-1))
            value = pd.cut(df[c], bins=rangeq[c]).unique()
            key = df[c+'Bin'].unique()
            dictState[c] = {key[i]: value[i] for i in range(len(key))}
            dfTest[c + 'Bin'] = pd.cut(dfTest[c], bins=rangeq[c], labels=range(0, len(rangeq[c]) - 1))


        print(dictState)
        # print(df2['tradingVolBin'].value_counts())
        # print(df2['tradingImbBin'].value_counts())
        # print(df2['spreadBin'].value_counts())
        # print(df2['movingDiffBin'].value_counts())
        # print(df2['OFIBin'].value_counts())

        # divide no. of shares into 5 intervals, for state variable
        rangeV = np.linspace(-lot, V, noLot + 1)
        rangeV_value = pd.cut(rangeV, noLot + 1)
        rangeV_key = pd.cut(rangeV, noLot + 1, labels=range(noLot + 1))
        dictState['vBin'] = {rangeV_key[i]: rangeV_value[i] for i in range(len(rangeV_key))}
        print(dictState)

        del df2, dfState, fig, rangeV, rangeV_key, rangeV_value, rangec, rangeq, regPImpact, corr, value, var


        #### TRAINING
        # define market state variables, may add 'OFIBin', 'microPriceBin' or 'bestBidVolBin', or change depending on stock
        marketState = ['tradingVolBin', 'tradingImbBin', 'movingDiffBin', 'spreadBin', 'orderImbBin']
        epiListTrain = df['epi'].unique()
        epiListTest = dfTest['epi'].unique()

        # initialize q function
        # list of dimension of state, excluding time step
        rangeS = []
        for m in marketState:
            n = m[:(len(m) - 3)]
            rangeS.append(len(dictState[n]))

        actionSpace = np.arange(0, V + lot, lot)

        # q_func has dimension of state space x 2 type of actions x action space
        q_func = np.full(rangeS + [int(noStep), noLot + 1, 2, actionSpace.shape[0]], np.nan)

        # for convergence plot, q_func has dimension of state space x 2 type of actions x action space x trainSize
        # set q_func2 = 0 and comment line 57 in tabular Q-learning function if dont want to plot convergence
        q_func2 = np.full(rangeS + [int(noStep), noLot + 1, 2, actionSpace.shape[0], len(epiListTrain)], np.nan)

        # for each learning rate
        for a in [0.1]:
            c = 0

            for i in tqdm(epiListTrain):
                if df[df['epi'] == i].shape[0] > 1:
                    run_episode(q_func, q_func2, df[df['epi'] == i], dictState, marketState, a, for_training=True, c=c)
                    c += 1

            for i in tqdm(epiListTest):
                if dfTest[dfTest['epi'] == i].shape[0] > 1:
                    optimalQ, optimalAType, optimalA, accReward = run_episode(q_func, q_func2, dfTest[dfTest['epi'] == i],
                                                                              dictState, marketState, a, for_training=False)
                    dfTest.loc[dfTest['epi'] == i, 'optimalQ' + str(a)] = optimalQ
                    dfTest.loc[dfTest['epi'] == i, 'optimalAType' + str(a)] = optimalAType
                    dfTest.loc[dfTest['epi'] == i, 'optimalA' + str(a)] = optimalA
                    dfTest.loc[dfTest['epi'] == i, 'accReward' + str(a)] = accReward

            # store as csv files
            outputFileh5 = outputFolder + "\\" + "BestPrice" + str(a) + str(BETA) + ".h5"
            h5f = h5py.File(outputFileh5, 'w')
            h5f.create_dataset(s, data=q_func2)
            h5f.close()

        # store as csv files
        outputFile = outputFolder + "\\" + str(BETA) + "BestPrice" + ".csv"
        dfTest.to_csv(path_or_buf=outputFile, index=False)

        del q_func, q_func2, df, dfTest




