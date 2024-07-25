import pandas as pd
import numpy as np

df = pd.read_csv('./data/TGR5agonists.tsv', sep='\t')

# データ前処理
df = df.dropna(subset=['EC50 (nM)'])
df = df[~df['EC50 (nM)'].str.contains('<|>')]

"""
データ標準化
EC50は小さければ小さいほど活性が高いので、大きい値になるように変換
"""
df['EC50 (nM)'] = df['EC50 (nM)'].astype(float)
df['EC50 (nM)'] = 1 / df['EC50 (nM)']

from sklearn.preprocessing import StandardScaler

df['EC50 (nM)'] = StandardScaler().fit_transform(df['EC50 (nM)'].values.reshape(-1, 1))


# maccs keys作成
from rdkit import Chem
from rdkit.Chem import AllChem

smiles_list = df['Ligand SMILES'].tolist()
maccs = [AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
maccs_df = pd.DataFrame(np.array(maccs, int))
# ecfp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 1024) for smiles in smiles_list]
# ecfp_df = pd.DataFrame(np.array(ecfp, int))

# 学習
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = maccs_df
y = df['EC50 (nM)'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# 性能評価、可視化
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('True')
plt.ylabel('Predict')

plt.show()


