import pandas as pd
df = pd.read_csv('./data/TGR5agonists.tsv', sep='\t')

"""
データ前処理
- 欠損値の削除
- EC50が正確な値でないもの(<, >)の削除
"""
df = df.dropna(subset=['EC50 (nM)'])
df = df[~df['EC50 (nM)'].str.contains('<|>')]


