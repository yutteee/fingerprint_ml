"""
モデルによる学習を行う
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def trainModel(X_train, X_test, y_train, y_test):
    # TODO: ハイパーパラメータのチューニング
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)

    with open('./model/rf.pkl', 'wb') as f:
        pickle.dump(rf, f)

    y_pred = rf.predict(X_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    trainModel()