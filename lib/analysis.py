import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; 
sns.set()

def find_biggest_full_frame(asset_, df):
    '''
    returns df without nan values for one specific asset
    '''
    # filter only rows that are not null
    target_asset_notnull = df[asset_].notnull()
    # apply filter
    df_w_clean_target = df[target_asset_notnull]
    # define maximum allowed nan values per asset
    nan_count_tresh = 0
    # filter only rows that have less than the nan_count_tresh nan's
    mask_mean_nan_cols = df_w_clean_target.isnull().sum(axis=0) <= nan_count_tresh
    # apply filter on dataframe
    df_w_clean_target = df_w_clean_target.loc[:, (df_w_clean_target.isnull().sum(axis=0) <= mask_mean_nan_cols)]
    return df_w_clean_target

def biggest_frames(df_filtered_futures):
    '''
    create and return table with futures and its biggest frame    
    each row represents one future with its amount of days and other futures
    return table with cols days and futures
    '''
    count_days = []
    count_commodities = []

    for commodity_ in df_filtered_futures.columns:
        for_commodity = find_biggest_full_frame(commodity_, df_filtered_futures)
        count_days.append(for_commodity.shape[0])
        count_commodities.append(for_commodity.shape[1])
        #print("commodity: ",commodity_, " has data for: " ,for_commodity.shape[0],
        #      " days and ", for_commodity.shape[1], " commodities")

    df = pd.DataFrame([count_days, count_commodities])
    df = df.transpose()
    df.columns = ['days', 'futures']
    return df

def scatter_plot_biggest_frames(df):
    '''
    takes df from biggest_frames function
    '''
    fontsize=18
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlabel('days',size=fontsize)
    ax.set_ylabel('futures',size=fontsize)
    ax.set_title("biggest full frame for every future", size=fontsize)

    ax.annotate('one dot represents the size of one future matrix [x-days,y-futures] \n having full data (no nan-values) for the widest period',
                xy=(550, 450), xytext=(580, 1000),
                arrowprops=dict(facecolor='grey', shrink=0.05), size=fontsize,)
    ax = sns.scatterplot(x="days", y="futures", s=250, data=df)