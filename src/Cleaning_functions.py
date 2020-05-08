import pandas as pd

def categorize(columns, values, ordered=True):
    df = pd.DataFrame()
    for c, v in zip(columns, values):
        df[f'{c.name}_cat'] = pd.Categorical(c, ordered = ordered, categories = v).codes
    return df