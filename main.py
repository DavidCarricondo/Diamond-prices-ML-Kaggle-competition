import pandas as pd
import src.Cleaning_pipeline as pn
from sklearn.ensemble import RandomForestRegressor

#Importing data
train = pd.read_csv('./INPUT/diamonds_train.csv') 
test = pd.read_csv('./INPUT/diamonds_test.csv')

def main():
    
    ##Cleaning
    train_clean = pn.clean_data(train, test_data=False)
    test_clean = pn.clean_data(test, test_data=True)


    X = train_clean.drop(['price'], axis= 1)
    y = train_clean.price

    print("Fittin the model...")
    mod = RandomForestRegressor(n_estimators=700, criterion='mse', max_depth=None, min_samples_split=6, min_samples_leaf=2, random_state=0)
    mod.fit(X,y)


    ##PREDICTING

    y_pred = mod.predict(test_clean)

    ##Exporting
    result = pd.DataFrame({'id':test_clean.index, 'price':y_pred})
    result.to_csv('./OUTPUT/submission.csv', index=False)

    print("Your submission is ready in './OUTPUT/submission.csv'")

if __name__=='__main__':
    main()