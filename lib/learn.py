import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def get_rf_df_returns():
    '''
    return dataframe for running rf
    '''
    returns = pd.read_csv("./data_clean/CME_futures_all_clean_backup.csv")
    returns.columns = returns.columns.str.replace('-Last', '')
    returns = returns.drop(['Date'], axis=1)
    return returns

def get_clusters():
    '''
    load cluster indexes
    return dict with dominant asset as key and its values as a list object
    remove clusters smaller than 5
    '''
    clusters = np.load('./numpy_data/clustering_and_representatives_full.npy').item()
    keys = list(clusters.keys())

    for i in keys:
        if len(clusters[i]) < 20:
            if i in clusters:
                del clusters[i]

    keys = list(clusters.keys())
    return clusters

def show_cluster_stats(returns, clusters):
    '''
    just to know which assets we have 
    in total we work with 714 futures as RF-features
    '''
    assets_to_predict = ["CME_CY10", "CME_FO11", "CME_S6", "CME_NP17", "CME_TY1", "CME_HP14"]
    assets_to_predict_num = [504, 945, 2629, 2198, 2726, 1354]
    for key in assets_to_predict_num:
        asset_to_predict = returns.columns[key]
        print("index:", key, "representative asset:", returns.columns[key], ": has nr of childs:", len(clusters[key]))

def get_one_prediction(returns, from_, to_, asset_, predictions=1, trees=100):
    '''    
    X - train features (multiple col - features)
    Y - train labels (single col - target)
    '''
    # sample data
    X = returns.iloc[from_:to_].dropna(axis=1)
        
    #print(X.columns)
    # the value we want to predict // shift down -1, iloc -1 (last will be nan after shift) shape is one smaller than sample_data
    Y = X[asset_].shift(-predictions).iloc[:-predictions]
    
    # the last value we want to predict from the predictors 
    lastpredictors = X.tail(predictions)
    
    # cut away the last one from the predictors to use as training
    X = X.iloc[:-predictions]
    
    # model forrest with 100 decision-trees, n_jobs for max speed & train the model
    rf = RandomForestRegressor(n_estimators = trees, n_jobs = -1)#, random_state = 123) 
    rf.fit(X, Y) 
    
    # prediction for the day(s) that we cut away from the predictor data
    predictions = rf.predict(lastpredictors)
    
    #print("features: ",X.shape, Y.shape)
    return predictions


def draw_predictions(returns, predictions, asset_to_predict, from_, to_, w):
    '''
    return pyplot showing predicted and original values of one specific asset
    '''
    original = returns.iloc[from_+w : to_+w][asset_to_predict]

    df = pd.DataFrame(original)
    df[asset_to_predict+"_pred"] = predictions
    fontsize=14
    fig, ax = plt.subplots(figsize=(16,5))
    ax.set_title("predictions for "+asset_to_predict, size=fontsize)

    ax.set_ylabel('returns',size=fontsize)
    ax.set_xlabel('days',size=fontsize)

    plt.plot(df)
    ax.legend(df.columns.values)
    plt.show()
    

def run_rf_on_selected_clusters(returns, clusters, trees):
    '''
    run for each representative asset a random forrest algorithm 
    with its cluster-childs as feature parameter
    '''
    assets_to_predict_for_numbers = [504, 945, 2629, 2198, 2726, 1354]

    for key in assets_to_predict_for_numbers:
        # target asset
        asset_to_predict = returns.columns[int(key)]

        # df with no na's in asset to predict
        df_asset_to_predict = returns[returns[asset_to_predict].notnull()].replace([np.inf, -np.inf], np.nan)
        df_asset_to_predict = df_asset_to_predict[df_asset_to_predict[asset_to_predict].notnull()]

        # df contains only the target cluster childs (features)
        df_asset_to_predict = df_asset_to_predict[returns.columns[clusters[key]]]
        df_asset_to_predict.reset_index(drop=True, inplace=True)

        # nr of rows
        asset_length = df_asset_to_predict.shape[0]

        # windowsize # 100
        w = int(df_asset_to_predict.shape[1]/5)
        if w > 100:
            w = 100

        # start 0 (+w)
        range_from = 0
        # -w because otherwise it will predict w-times into the future / maximal den nächsten Tag voraussagen sonst gehen die features aus -> ValueError 
        range_to = asset_length-w#+2

        #####
        print("Asset to predict", asset_to_predict, "df", df_asset_to_predict.shape, "Window", w, "from", range_from, "to", range_to, "trees", trees)
        #####

        rf_path = './results/'+asset_to_predict
        df_asset_to_predict.to_csv(rf_path+'_returns.csv')
        asset_stats = [range_from, range_to, w]
        np.save(rf_path+'_stats.npy', asset_stats)

        predictions_raw = []    
        pred_compare = []
        pred_labels = []

        for current_from in range(range_from, range_to):
            current_to = current_from + w

            # return single unprocessed price return prediction(s)
            preds = get_one_prediction(df_asset_to_predict, current_from, current_to, asset_to_predict, 1, trees)
            predictions_raw.append(preds[0])  

            # label the next day if positive or negative for each sample
            pred_single_label = np.sign(preds[0])
            pred_labels.append(pred_single_label)

            # calc for t1
            g_t = pred_single_label * df_asset_to_predict[asset_to_predict].iloc[current_to]
            pred_compare.append(g_t)

            print(current_to, preds, pred_single_label, g_t)

        print("Asset predicted successfully, saving its values..")

        np.save(rf_path+'_return_predictions.npy', predictions_raw)
        np.save(rf_path+'_prediction_values.npy', pred_compare)
        np.save(rf_path+'_label_values.npy', pred_labels)

def get_score(returns, predictions, asset_to_predict, from_, to_, w):
    '''
    print different accuracy scores
    '''
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    original = returns.iloc[from_+w : to_+w][asset_to_predict]

    df = pd.DataFrame(original)
    df[asset_to_predict+"_pred"] = predictions

    df = np.sign(df)
    y_true = df[asset_to_predict].tolist()
    y_pred = df[asset_to_predict+'_pred'].tolist()

    print("Accuracy Score         ", accuracy_score(y_true, y_pred))
    print("F1 Score (avg=macro)   ", f1_score(y_true, y_pred, average='macro'))
    print("F1 Score (avg=micro)   ", f1_score(y_true, y_pred, average='micro'))
    print("F1 Score (avg=weighted)", f1_score(y_true, y_pred, average='weighted'))
    print("F1 Score (avg=none)    ", f1_score(y_true, y_pred, average=None))
    

def show_rf_result_stats(returns):
    assets_to_predict_num = [504,945,2629,2198,2726,1354]
    assets_to_predict = ["CME_CY10", "CME_FO11", "CME_S6", "CME_NP17", "CME_TY1", "CME_HP14"]
    
    for key in assets_to_predict:
        asset_to_predict = key
        rf_path = './results/'+asset_to_predict
        df_asset_to_predict = pd.read_csv(rf_path+'_returns.csv')
        pred_compare = np.load(rf_path+'_prediction_values.npy').tolist()
        range_from = np.load(rf_path+'_stats.npy').tolist()[0]
        range_to = np.load(rf_path+'_stats.npy').tolist()[1]
        w = np.load(rf_path+'_stats.npy').tolist()[2]

        draw_predictions(df_asset_to_predict, pred_compare, asset_to_predict, range_from, range_to, w)
        get_score(df_asset_to_predict, pred_compare, asset_to_predict, range_from, range_to, w)
        
        
def plot_single_future(returns, asset_name):
    import matplotlib.pyplot as plt
    asset = returns[asset_name]
    fontsize=14
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_xlabel('days',size=fontsize)
    ax.set_ylabel('returns',size=fontsize)
    ax.set_title(asset_name, size=fontsize)
    plt.plot(asset)