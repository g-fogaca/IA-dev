# importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt

# importando dataset
df = pd.read_csv("housing.csv")
df.info()
features = df.values

#%%
# visualizando
plt.figure(figsize=(5,5))
plt.style.use("ggplot")
plt.scatter(df["Longitude"], df["Latitude"], alpha=0.5, c=df["MedInc"])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

#%%
# normalizando os dados
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(features)
normalized_features = scaler.transform(features)

del scaler

#%%
# utilizando o modelo kmeans
from sklearn.cluster import KMeans

features_1 = normalized_features[:,[0,6,7]]

# testando número de clusters
inertias = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(features_1)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker="o")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.show()

del inertias, k, kmeans

#%%
for k in range(1,7):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(features_1)
    labels = kmeans.predict(features_1)
    
    df['Region'] = labels

    plt.figure(figsize=(5,5))
    plt.style.use("ggplot")
    plt.scatter(df["Longitude"], df["Latitude"], alpha=0.5, c=df["Region"])
    plt.title(f'{k} clusters')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

del k, kmeans, labels, features, features_1

#%%
df.boxplot("MedHouseVal", "Region")
plt.show()
