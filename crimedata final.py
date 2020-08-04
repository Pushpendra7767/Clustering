# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch
from sklearn.cluster import	AgglomerativeClustering 
# print dataset
cri = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\crimedata.csv")
cri.columns
ci=cri.columns
print(ci)
# apply normalization function on dataset.
def norm_func(i):
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
df_norm = norm_func(cri.iloc[:,1:])
df_norm.head()

# k means
k = list(range(2,15))
k
TWSS = []  
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
#  Scree plot
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)
model.labels_  
md=pd.Series(model.labels_)  
cri['clust']=md 
df_norm.head()
cri = cri.iloc[:,[5,0,1,2,3,4]]
cri.iloc[:,1:6].groupby(cri.clust).mean()
cri.to_csv("crimda.csv")
# Hierarchical clustering
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)
plt.show()
# AgglomerativeClustering
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 
cluster_labels=pd.Series(h_complete.labels_)
cri['clust']=cluster_label
cri.head()






























































