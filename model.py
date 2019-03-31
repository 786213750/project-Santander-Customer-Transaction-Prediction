import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc
from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
import gc

df=pd.read_csv("C:\\thomas\\project Santander\\data\\train.csv")
train_X=df.drop(["ID_code","target"],axis=1)
train_y=df["target"]
print("finish loading")
del df
gc.collect()
X_train, X_val, y_train, y_val =train_test_split(train_X,train_y,test_size=0.20, random_state=42)
X_train, X_val = csr_matrix(X_train, dtype='float32'), csr_matrix(X_val, dtype='float32')
print("finish split")
lgb_model = lgb.LGBMClassifier(max_depth=-1,
                               n_estimators=30000,
                               learning_rate=0.1,
                               num_leaves=2**12-1,
                               colsample_bytree=0.28,
                               objective='binary', 
                               n_jobs=-1)

lgb_model.fit(X_train, y_train, eval_metric='auc', 
              eval_set=[(X_val, y_val)], 
              verbose=100, early_stopping_rounds=100)

ax = lgb.plot_importance(lgb_model, max_num_features=10)
plt.show()