"""
変数が多く、過学習が起こりやすいため、変数選択を行う
bortaによる変数選択を行う
ランダムな特徴量を作成し、そのランダムな特徴量とオリジナルな特徴量をランダムフォレストで学習し、変数の重要度がランダムな特徴量よりも高いものを選択する
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

def selectFeature():
    df = pd.read_csv('./data/EC50.csv')
    fp_df = pd.read_csv('./data/fingerprint.csv')

    X = fp_df
    y = df['values'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # borutaによる変数選択
    corr_list = []
    for n in range(10000):
        shadow_features = np.random.rand(X_train.shape[0]).T
        corr = np.corrcoef(X_train, shadow_features,rowvar=False)[-1]
        corr = abs(corr[corr < 0.95])
        corr_list.append(corr.max())
        corr_array = np.array(corr_list)
        perc = 100 * (1-corr_array.max())

    rf = RandomForestRegressor(n_estimators=100, random_state=0)

    feat_selector = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=0,
        alpha=0.05,
        max_iter=50,
        perc=perc,
        random_state=0
    )

    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_

    feat_selector.fit(X_train.values, y_train.values)

    X_train_selected = X_train.iloc[:, feat_selector.support_]
    X_test_selected = X_test.iloc[:, feat_selector.support_]

    print(X_train_selected.shape)

    fp_df = fp_df.loc[:, fp_df.columns.isin(X_train_selected.columns)]
    df = pd.concat([df, fp_df], axis=1)
    df.to_csv('./data/inputData.csv', index=False)

    print("Feature selection is completed.")

    return {
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_test': y_test
    }

if __name__ == '__main__':
    variables = selectFeature()
    print(variables)