"""
データ前処理
- 欠損値削除
- EC50 (nM)の<, >を削除
- EC50は小さいほど強いので逆数を取り、標準化
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def preprocessing():
    df = pd.read_csv('./data/TGR5agonists.tsv', sep='\t')

    df = df.dropna(subset=['EC50 (nM)'])
    df = df[~df['EC50 (nM)'].str.contains('<|>')]

    df['EC50 (nM)'] = df['EC50 (nM)'].astype(float)
    # EC50の分布のせいで過学習してしまっている？
    # 500〜5000のデータを削除してみる
    # df = df[df['EC50 (nM)'] < 5000]
    # df = df[df['EC50 (nM)'] > 500]

    # df['EC50 (nM)'] = 1 / df['EC50 (nM)']
    # df['EC50 (nM)'] = StandardScaler().fit_transform(df['EC50 (nM)'].values.reshape(-1, 1))

    # 分布を確認
    plt.hist(df['EC50 (nM)'], bins=100, range=(0, 20000))
    plt.show()

    return df

if __name__ == '__main__':
    df = preprocessing()
    print(df.head())
    print(df.shape)