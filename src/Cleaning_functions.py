import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


def categorize(df, columns, values,remove=True, ordered=True):
    '''
    Convert categories into ordered numerical and removes old features if remove==True
    '''
    new_df = pd.DataFrame()
    for c, v in zip(columns, values):
        new_df[f'{c}_cat'] = pd.Categorical(df[c], ordered = ordered, categories = v).codes
        if remove:
            df.drop([c],axis=1,inplace=True )
    return pd.concat([df , new_df], axis=1, sort=False )


def logit(df, columns, remove=True):
    '''
    Convert columns to logarithmics and remove old features
    '''
    new_df = pd.DataFrame()
    for e in columns:
        name = e + '_log'
        new_df[name] = df[e].apply(lambda x: np.log1p(x))
    if remove:
        df_no_col = df.drop(columns, axis=1).reset_index(drop=True)
        return pd.concat([df_no_col, new_df], axis=1, sort=False)
    return pd.concat([df,new_df],axis=1, sort=False)

def remove_outliers(df, columns, threshold=3):
    '''
    Remove rows with outliers according to the z-score. Threshold is 3 by default
    '''
    for col in columns:
        z = np.abs(stats.zscore(df[col]))
        for i,e in enumerate(z):
            if e > threshold:
                df.drop(axis=0, index=i, inplace=True)
                df.reset_index()
        return df.reset_index()


def standardize(df, columns, remove=True):
    scaler = StandardScaler()
    scale_feat = scaler.fit_transform(df[columns])
    new_df = pd.DataFrame(scale_feat, columns=[c+'_st' for c in columns])
    if remove:
        df.drop(columns, axis=1, inplace=True)
    return pd.concat([df, new_df], axis=1, sort=False)


