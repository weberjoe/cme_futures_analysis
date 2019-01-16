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
    
    # only nan cols
    nullcolumns = df.isnull().all()
    
    # remove nan cols
    df = df.dropna(axis=1, how='all')
    
    return df

# choose interval in df with least nan values
def select_interval(df, selection):
    df_filtered = df[selection:]
    return df_filtered

# calculate log returns for dataframe
def calcLogreturns(df):
    returns = df.pct_change(1, fill_method=None)
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
        
def exchange_counter():
    from collections import Counter
    path="./data/"
    chris_meta=pd.read_csv(path+"CHRIS_metadata.csv") # loading data
    exchange=chris_meta["code"]                       # extracting code column 
    first_three_letters = exchange.str[:3]            # extracting first three columns of exchange
    # display(first_three_letters)
    counts = Counter(first_three_letters)

    labels, values = zip(*counts.items())
    indexes = np.arange(len(labels))
    width = 1
    plt.figure(1,(15,8))
    plt.bar(indexes, values, width)
    plt.xticks(indexes, labels)
    plt.xlabel("Exchange Code")
    plt.ylabel("Number of Futures")
    plt.show()

def summarise_CME():
    import quandl
    API_KEY_JOE="16-3ue4hzwtKNj3DSFYY"
    quandl.ApiConfig.api_key=API_KEY_JOE
    df_sample_futures = quandl.get("CHRIS/CME_B630", collapse="daily")
    path="./data/"
    all_CME = pd.read_csv(path+"CME_futures_all.csv", index_col=0)
    column_names=pd.DataFrame(list(df_sample_futures),columns=["CME Variables"])
    display(column_names)
    print("Number of Days in CME Dataset: ", all_CME.shape[:-1],"\n")
    print("Number of Futures in CME Dataset: ", all_CME.shape[-1:])