"""
フィンガープリント作成
これを特徴量として使う
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def generateFeature():
    # データ読み込み
    df = pd.read_csv('./data/EC50.csv')

    smiles_list = df['Smiles'].tolist()
    ecfp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 4, 4096) for smiles in smiles_list]
    fp_df = pd.DataFrame(np.array(ecfp, int))

    # 分散が0の変数を削除
    fp_df = fp_df.loc[:, fp_df.var() != 0]

    # 相関係数が高い変数を削除 -> 一つの構造による重複した寄与を削除
    from dcekit.variable_selection import search_highly_correlated_variables
    threshold_of_r = 0.95 #変数選択するときの相関係数の絶対値の閾値
    corr_var = search_highly_correlated_variables(fp_df, threshold_of_r)
    fp_df.drop(fp_df.columns[corr_var], axis=1, inplace=True)

    # 5分子以上で出現するフィンガープリントのみを残す
    fp_df = fp_df.loc[:, fp_df.sum() >= 5]

    # csvに保存
    fp_df.to_csv('./data/fingerprint.csv', index=False)

    print("Feature generation is completed.")

    return fp_df

if __name__ == '__main__':
    df = generateFeature()
    print(df)