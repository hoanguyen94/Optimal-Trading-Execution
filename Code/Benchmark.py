#!/usr/bin/env python
# coding: utf-8

# In[1]:


from DataAnalysis import *


inputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Output'    #'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Simulation\\Data'
# uncomment the below line if want to test specific stocks
stock = ['CATY']

# create output folder for all stocks

def immReward(df, iniPrice, time, v, PI=0):
    """ Immediate rewards from market order """
    reward = (df['bestBidPrice'].iloc[time] - iniPrice + PI) * min(v, df['bestBidVol'].iloc[time]) * (GAMMA ** time)
    reward += (df['bestBidPrice'].iloc[time] - iniPrice + PI) * max(v - df['bestBidVol'].iloc[time], 0) * (GAMMA ** time)
    return reward


for s in stock:
    sInputFolder = inputFolder + '\\' + s + '\\Training'
    df = pd.read_csv(sInputFolder + '\\' + '0BestPrice.csv')
    df = df[df['optimalQ0.1'].notnull()]

    epiList = df['epi'].unique()

    # initialize columns
    df.loc[:, ['rewardSL', 'rewardTWAP']] = np.nan
    df.loc[:, 'sharesFilledSL'] = 0

    for e in tqdm(epiList):
        if df[df['epi'] == e].shape[0] > 1:
            df3 = df[df['epi']==e]
            iniPrice = df3['midPrice'].iloc[0]

            ### submit and leave
            # limit order is (partially) filled
            sharesFilled = min(V, df3['2ndFillRo'].iloc[0])
            rewardL = (df3['bestAskPrice'].iloc[0] - iniPrice) * sharesFilled
            vLimit = V - sharesFilled
            time = df3.shape[0] - 1
            rewardL += immReward(df3, iniPrice, time, vLimit)

            # save into df
            df.loc[(df['epi']==e) & (df['timeStep']==0), 'sharesFilledSL'] = sharesFilled
            df.loc[df['epi']==e, 'rewardSL'] = rewardL


            ### TWAP
            rewardM = 0
            vMarket = V
            PI = 0
            for t in range(df3.shape[0]):
                if t == df3.shape[0] - 1:
                    rewardM += immReward(df3, iniPrice, t, vMarket, PI)
                else:
                    rewardM += immReward(df3, iniPrice, t, lot, PI)

                PI = df3[['const', 'OFICoeff', 'TICoeff', 'OICoeff']].iloc[t] @ np.array([1, -lot, -lot, -lot])
                vMarket -= lot
            # save into df
            df.loc[df['epi']==e, 'rewardTWAP'] = rewardM

    # save df as csv file
    outputFolder = inputFolder + "\\" + s + "\\Benchmark"
    Path(outputFolder).mkdir(parents=True, exist_ok=True)
    outputFileS = outputFolder + "\\BenchmarkAdded.csv"
    df.to_csv(path_or_buf=outputFileS, index=False)


