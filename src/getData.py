"""
データ取得
ChEMBLとBindingDBのデータを取得し、EC50の値を取得する
EC50の値が70以下のものを1, 1000以上のものを0にする(だいたい上位、下位30%)
"""

import pandas as pd

def getData():
    # dbからデータを取得
    # EC50の値に範囲があるものは除外
    ChEMBL_df = pd.read_csv("./data/ChEMBL.tsv", sep='\t')
    ChEMBL_df = ChEMBL_df.dropna(subset=['Standard Relation'])
    ChEMBL_df = ChEMBL_df.dropna(subset=['Standard Value'])
    ChEMBL_df = ChEMBL_df[ChEMBL_df['Standard Relation'].str.contains("'='")]
    ChEMBL_df = ChEMBL_df[ChEMBL_df['Standard Units'].str.contains('nM')]

    BindingDB_df = pd.read_csv("./data/BindingDB.tsv", sep='\t')
    BindingDB_df = BindingDB_df.dropna(subset=['EC50 (nM)'])
    BindingDB_df = BindingDB_df[~BindingDB_df['EC50 (nM)'].str.contains('<|>')]
    BindingDB_df['EC50 (nM)'] = BindingDB_df['EC50 (nM)'].astype(float)

    # 両データフレームを結合
    ChEMBL_df = ChEMBL_df[['Smiles', 'Standard Value']]
    ChEMBL_df = ChEMBL_df.rename(columns={'Standard Value': 'EC50 (nM)'})
    BindingDB_df = BindingDB_df[['Ligand SMILES', 'EC50 (nM)']]
    BindingDB_df = BindingDB_df.rename(columns={'Ligand SMILES': 'Smiles'})

    df = pd.concat([ChEMBL_df, BindingDB_df])

    # 重複を削除
    df = df.drop_duplicates(subset='Smiles')
    df.to_csv('EC50.csv', index=False)

    # 値が0のものを削除
    df = df[df['EC50 (nM)'] != 0]

    # EC50が100以下のものを1, 1000以上のものを0にし、それ以外を削除
    df.loc[df['EC50 (nM)'] <= 70, 'values'] = 1
    df.loc[df['EC50 (nM)'] >= 1000, 'values'] = 0
    df = df[(df['values'] == 1) | (df['values'] == 0)]

    # csvに保存
    df.to_csv('./data/EC50.csv', index=False)

    print("Data acquisition is completed.")

    return df

if __name__ == '__main__':
    df = getData()
    print(df)