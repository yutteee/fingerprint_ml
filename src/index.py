import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/TGR5agonists.tsv', sep='\t')

"""
データ前処理
- 欠損値削除
- EC50 (nM)の<, >を削除
- EC50は小さいほど強いので逆数を取り、標準化
"""
from sklearn.preprocessing import StandardScaler

df = df.dropna(subset=['EC50 (nM)'])
df = df[~df['EC50 (nM)'].str.contains('<|>')]

df['EC50 (nM)'] = df['EC50 (nM)'].astype(float)
# EC50の分布のせいで過学習してしまっている？
# 500〜5000のデータを削除してみる
# df = df[df['EC50 (nM)'] < 5000]
# df = df[df['EC50 (nM)'] > 500]

# EC50の逆数を取り、標準化
df['EC50 (nM)'] = 1 / df['EC50 (nM)']
df['EC50 (nM)'] = StandardScaler().fit_transform(df['EC50 (nM)'].values.reshape(-1, 1))

"""
フィンガープリント作成
"""
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Sheridan

smiles_list = df['Ligand SMILES'].tolist()
maccs = [AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
fp_df = pd.DataFrame(np.array(maccs, int))
ecfp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 4, 4096) for smiles in smiles_list]
fp_df = pd.concat([fp_df, pd.DataFrame(np.array(ecfp, int))], axis=1)

# 分散が0の変数を削除
fp_df = fp_df.loc[:, fp_df.var() != 0]

# 相関係数が高い変数を削除 -> 一つの構造による重複した寄与を削除
from dcekit.variable_selection import search_highly_correlated_variables
threshold_of_r = 0.5 #変数選択するときの相関係数の絶対値の閾値
corr_var = search_highly_correlated_variables(fp_df, threshold_of_r)
fp_df.drop(fp_df.columns[corr_var], axis=1, inplace=True)

"""
学習
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier

X = fp_df
y = df['EC50 (nM)'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# borutaによる変数選択
from boruta import BorutaPy

rf = RandomForestRegressor(n_estimators=100, random_state=40)

feat_selector = BorutaPy(
    rf,
    n_estimators='auto',
    verbose=0,
    alpha=0.05,
    max_iter=50,
    perc=100,
    random_state=0
)

feat_selector.fit(X_train, y_train)

X_train_selected = X_train[:, feat_selector.support_]
X_test_selected = X_test[:, feat_selector.support_]

# モデルの学習
rf.fit(X_train_selected, y_train)

# 性能評価、可視化
y_pred = rf.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

plt.scatter(y_test, y_pred)
plt.xlabel('True')
plt.ylabel('Predict')

plt.show()


