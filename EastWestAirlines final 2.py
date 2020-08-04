# import pacakegs
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
air = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Clustering\\EastWestAirlines.csv")
air.columns
ai=air.columns
print(ai)

# apply normalization function on dataset.
def norm_func(i):
    x = (i-i.min())	/ (i.max() - i.min())
    return (x)
df_norm = norm_func(air.iloc[:,1:])
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
air['clust']=md 
df_norm.head()
air = air.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
air.iloc[:,1:7].groupby(air.clust).mean()
air.to_csv("eastair.csv")


# Hierarchical clustering
# apply normalization function on dataset.
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
df_norm = norm_func(air.iloc[:,1:])
type(df_norm)

# Hierarchical clustering.
# p = np.array(df_norm) # converting into numpy array format 
# plot dendogram  # linkage method = complete, distance method = euclidean
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)
plt.show()

# plot dendogram  # linkage method = average, distance method = cosine
z1 = linkage(df_norm, method="average",metric="cosine")
plt.figure(figsize=(20, 5));plt.title('Hierarchical Clustering Dendrogram 1');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z1,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)
plt.show()

# plot dendogram  # linkage method = median, distance method = euclidean
z2 = linkage(df_norm, method="median",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram 2');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z2,
    leaf_rotation=0.,  
    leaf_font_size=8.,  
)
plt.show()


































































