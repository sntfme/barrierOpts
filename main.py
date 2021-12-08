import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

pd.set_option('display.max_columns', None)

# Getting daily adjusted close prices
tickers = 'nvda amd intc'                                   # accepts any number of tickers as space delimited string
dfs = [yf.download(tickers=tickers, \
                   start=(datetime.today().date() - i*timedelta(days=365)),\
                   end=datetime.today().date(), \
                   interval='1d')\
           ['Adj Close'].reset_index(drop=True) for i in range(1, 6)]

# Init
stdevs = [np.array(dfs[i].std()) for i in range(len(dfs))]  # stdevs for each ticker for each tenor
starts = np.array(dfs[0])[-1]                               # start prices for simulation
r = 0.02                                                    # risk free rate
dt = 1./252.                                                # one day step
num_sims = 1000                                             # number of simulations

# MC Simulation
simulations = []
for sim in range(num_sims):
    finals = []
    for year in range(len(dfs)):
        for i in range(len(starts)):
            S = starts[i]
            for day in range((1+year)*252):
                dS = (r*dt + stdevs[year][i]*np.sqrt(dt)*np.random.normal(0, 1))
                S += dS
            finals.append(S)
    simulations.append(finals)

# Raising flags, flag == 1 denotes client receives coupon
cols = np.array(list(dfs[0].columns)*len(dfs), dtype=object)\
       + np.array([[f'_{i}y']*len(starts) for i in range(1,len(dfs)+1)], dtype=object).reshape(-1)

df = pd.DataFrame(simulations, columns=cols)

conditions = []
for j in range(0, len(dfs)*len(starts), 3):
    condition = []
    for i in range(len(starts)):
        c = df.iloc[:, i+j] > starts[i]
        condition.append(c)
    conditions.append(condition)

df['flag_1y'] = np.where(np.all(conditions[0], axis=0), 1, 0)
df['flag_2y'] = np.where(np.all(conditions[1], axis=0), 1, 0)
df['flag_3y'] = np.where(np.all(conditions[2], axis=0), 1, 0)
df['flag_4y'] = np.where(np.all(conditions[3], axis=0), 1, 0)
df['flag_5y'] = np.where(np.all(conditions[4], axis=0), 1, 0)
df['flags'] = df[['flag_1y','flag_2y','flag_3y','flag_4y','flag_5y']].sum(axis=1)

# Submission
dummy_df = pd.DataFrame(data={'col1':[0,1,2,3,4,5]})
probs = df['flags'].value_counts(normalize=True)
probs = probs.reindex_like(dummy_df).fillna(0)

sub_df = pd.DataFrame(data={
    'Payoff at Termination': ['100%','110%','120%','130%','140%','150%'],
    'Probability': probs})

print(f'\n After {num_sims} MC simulations we arrive at the following probabilities of payoffs:')
print(sub_df)