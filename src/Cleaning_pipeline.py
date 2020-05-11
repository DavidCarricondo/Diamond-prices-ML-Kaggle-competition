import pandas as pd
import src.Cleaning_functions as fn


def clean_data(df, test_data = False):
    #Transform to logarithmic
    features = ['carat','table','depth', 'x', 'y', 'z'] 

    df_clean = fn.logit(df, features)

    #Categories to numericals:
    #clarity = ('IF', 'VVS2','VS1','VS2','SI1','SI2','I1' )
    #color = ('D', 'E', 'F', 'G', 'H', 'I','J')
    #cut = ('Premium', 'Ideal', 'Very Good', 'Good','Fair')

    #df_clean = fn.categorize(df_clean, ['cut', 'color', 'clarity'], [cut, color, clarity])

    #Custom categories:

    df_clean.replace({'Premium': 1, 'Ideal': 2, 'Very Good':3, 'Good':4, 'Fair':5}, inplace=True)
    df_clean.replace({'D':1, 'E':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7}, inplace=True)
    df_clean.replace({'IF':1, 'VVS1':2, 'VVS2':3, 'VS1':4, 'VS2':5, 'SI1':6, 'SI2':7, 'I1':8}, inplace=True)

    #df_clean = df_clean.reset_index()
    
    #Categories to dummies:
    #df_clean = pd.get_dummies(df_clean, prefix=['cut','color', 'clarity'])
    
    #if test_data == False:
        #Remove outlayers:
        #columns = ['carat_log','x_log','y_log','z_log']

        #fn.remove_outliers(df_clean, columns)

    #Standardize columns:
    df_clean = fn.standardize(df_clean,['carat_log','table_log','x_log','y_log','z_log'] )
    #Without log transforming:
    #df_clean = fn.standardize(df_clean,['carat','depth','table','x','y','z'] )

    #Unuseful columns
    if 'index' in df_clean.columns:
        df_clean.drop(['index'], axis= 1, inplace=True)
    df_clean.drop(['Unnamed: 0','depth_log'], axis= 1, inplace=True) 

    return df_clean
    
    