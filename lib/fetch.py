import quandl
import numpy as np
import pandas as pd


def preprocess_chris_data(regex=False):
    '''
    use: futures_to_fetch_list = preprocess_chris_data(True)
         print("Furutes count: ",len(futures_to_fetch_list))
    if regex true only consider unique futures
    as each future has multiple similar contracts
    '''
    path="./data/"
    
    # load metadata about futures
    chris_meta=pd.read_csv(path+"CHRIS_metadata.csv")

    # Filter for market
    EXCHANGE="CME"
    chris_meta=chris_meta[chris_meta["code"].str.contains(EXCHANGE)]
    
    futures_to_fetch = []
    if regex:
        # apply regex to filter unique futures
        regex_="\D*[0-9]*\D+(0|[2-9])*1$"
        chris_meta = chris_meta[chris_meta["code"].str.match(regex_)]
        futures_to_fetch=chris_meta["code"]
        futures_to_fetch_list=futures_to_fetch.tolist()
    else:
        # without regex using all future contracts
        futures_to_fetch=chris_meta["code"]
        futures_to_fetch_list=futures_to_fetch.tolist()
        
    return futures_to_fetch.tolist()

def fetch_chris_from_quandl():
    '''
    use: df_futures = fetch_chris_from_quandl()
         df_futures.to_csv(path+EXCHANGE+"_commodities_all.csv")
    fetch and return dataframe of selected futures from quandl
    '''
    API_KEY_JOE="16-3ue4hzwtKNj3DSFYY"
    quandl.ApiConfig.api_key=API_KEY_JOE
    CHRIS="CHRIS/"
    COL_=".4"
    time_range=["daily","weekly","monthly","quarterly","annual"]
    
    # get list of futures to fetch from CHRIS file
    futures_to_fetch = preprocess_chris_data(True)
    sample_futures = futures_to_fetch[0:]
    sample_futures=[CHRIS + f + COL_ for f in sample_futures]
    df_futures = quandl.get(sample_futures, collapse=time_range[0])
    return df_futures
