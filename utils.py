import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
warnings.filterwarnings('ignore')
def julei():
    with open('hdim.pkl','rb') as f:
        hdim=pickle.load(f)
    return hdim

def silhouette_plot(codes, max_k=10):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    Scores = []  # 存silhouette scores
    for k in range(2, max_k):
        estimator = KMeans(n_clusters=k, random_state=555)  # construct estimator
        estimator.fit(codes)
        Scores.append(
            silhouette_score(codes, estimator.labels_, metric='cosine'))
    X = range(2, max_k)
    plt.figure()
    plt.xlabel('Cluster num K', fontsize=15)
    plt.ylabel('Silhouette Coefficient', fontsize=15)
    plt.plot(X, Scores, 'o-')
    plt.savefig('silhouette width.pdf')
    plt.savefig('silhouette width.png')
    return Scores

if __name__=='__main__':
    # hdim=julei()
    # pinjie=np.hstack(hdim)
    # print(silhouette_plot(pinjie))
    tpm=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/tpm.csv',index_col=0)
    mirna=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/mirna.csv',index_col=0)
    methy=pd.read_csv('/home/zhoulin/shiyan/dst_subtype/methy.csv',index_col=0)
    tpm=tpm.transpose()
    mirna=mirna.transpose()
    methy=methy.transpose()
    result_df_column = pd.concat([tpm, mirna, methy], axis=1)
    print(silhouette_plot(result_df_column.values).max())