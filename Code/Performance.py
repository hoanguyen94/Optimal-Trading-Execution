#!/usr/bin/env python
# coding: utf-8

# In[1]:


from DataAnalysis import *

inputFolder = 'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Output'    #'C:\\Users\\I535982\\Documents\\Uni\\Thesis\\Data\\Simulation\\Data'
# uncomment the below line if want to test specific stocks
# stock = ['CATY']   #['ANIP', 'ACGL', 'HSIC', 'BRKS', 'CORE', 'CBSH', 'CRUS', 'JCOM', 'CVLT', 'BJRI', 'ALLK', 'ALRM', 'AFYA', 'BOKF']

for s in stock:
    sInputFolder = inputFolder + '\\' + s
    df2 = pd.read_csv(sInputFolder + '\\Benchmark\\BenchmarkAddedTest.csv')

    # create state
    df2['state'] = df2['tradingVolBin'].apply(str) + df2['tradingImbBin'].apply(str) + df2['movingDiffBin'].apply(str) + df2['spreadBin'].apply(str) + df2['orderImbBin'].apply(str) + df2['timeStep'].apply(int).apply(str)

    # save df as test set
    df = df2[df2['optimalQ0.1'].notnull()]

    # plot performance
    df['epi'] = df['epi'] - df['epi'].iloc[0] + 1
    # for original data
    meanR1 = df['accReward1'].mean()
    meanR01 = df['accReward0.1'].mean()
    meanRSL = df['rewardSL'].mean()
    meanRTWAP = df['rewardTWAP'].mean()
    print(f'Stock {s}: Mean of: Observed optimal policies: {meanR1}, learned policies: {meanR01},'
          f'submit and leave: {meanRSL}, TWAP: {meanRTWAP}')

    varR01 = (df['accReward0.1'] - df['accReward1']).var()
    varRSL = (df['rewardSL'] - df['accReward1']).var()
    varRTWAP = (df['rewardTWAP'] - df['accReward1']).var()
    print(f'Stock {s}: var of learned policies: {varR01} compared to observed optimal policies, '
          f'submit and leave: {varRSL}, TWAP: {varRTWAP}')

    # Sharpe Ratio
    x = math.ceil(max(abs(meanR01), abs(meanRSL), abs(meanRTWAP)))
    R01 = (x + meanR01) / np.sqrt(varR01)
    RSL = (x + meanRSL) / np.sqrt(varRSL)
    RTWAP = (x + meanRTWAP) / np.sqrt(varRTWAP)
    print(f'Stock {s}: Sharped Ratio of learned policies: {R01}, submit and leave: {RSL}, TWAP: {RTWAP} ')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.lineplot(data=df[['epi', 'accReward1', 'accReward0.1']].groupby(['epi']).mean(), ax=axes[0]).set_title("Model 1")
    sns.lineplot(data=df[['epi', 'accReward1', 'rewardSL']].groupby(['epi']).mean(), ax=axes[1]).set_title("Submit and Leave")
    sns.lineplot(data=df[['epi', 'accReward1', 'rewardTWAP']].groupby(['epi']).mean(), ax=axes[2]).set_title("TWAP")
    plt.savefig(sInputFolder + '\\Benchmark\\Performance.png')
    plt.close()

    fig, axes = plt.subplots(1, figsize=(10, 10))
    sns.kdeplot(data=df[['epi', 'accReward1', 'accReward0.1', 'rewardSL', 'rewardTWAP']].groupby(['epi']).mean(), ax=axes).set_title("KDE")
    plt.savefig(sInputFolder + '\\Benchmark\\KDE.png')
    plt.close()

    print('\n')




