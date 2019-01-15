import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set()

def clean(df):
    '''
    # cleaning the dataframe
    # selection defines the day from which on we take data
    # run only once after loading data
    returns clean price returns dataframe
    '''
    # remove whitespace from cols
    df.columns = df.columns.str.replace(' ', '')
    df.columns = df.columns.str.replace('CHRIS/', '')
    #calculate log returns
    df = calcLogreturns(df)
    # original size
    print("raw: ", df.shape)
    # only nan cols
    nullcolumns = df.isnull().all()
    print("futures containing only nan: ", nullcolumns.sum())
    # remove nan cols
    df = df.dropna(axis=1, how='all')
    print("cleaned: ", df.shape)
    return df

# choose interval in df with least nan values
def select_interval(df, selection):
    df_filtered = df[selection:]
    return df_filtered

# calculate log returns for dataframe
def calcLogreturns(df):
    returns = df.pct_change(1, fill_method=None) # np.log(df.pct_change(1, fill_method=None)+1).replace
    return returns

def get_Total_NaN_of_df(df, having=False):
    if having:
        return (~asset_rolling.isna()).sum().sum()
    else:
        return df.isna().sum().sum()
    
def missingValues(df, from_=0, percentage=False):
    # nan percentage
    if percentage:
        count_nulls = df.isnull().sum(axis=1)/len(df.columns)
    else:
        count_nulls = df.isnull().sum(axis=1)
    plt.figure(figsize=((20,10)));
    plt.xticks([0,2000,4000,6000,8000,9000,10000,11000,12000,13000,14000,15000])
    count_nulls[from_:].plot(subplots=True, label='NaN Values')
    plt.legend()
    plt.xticks(rotation=25)
    plt.ylabel('percentage of nan values')
    plt.show()
    print("NaNs in df: ", get_Total_NaN_of_df(df))