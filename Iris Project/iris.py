# importando bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importando e visualizando dados
df = sns.load_dataset("iris")
sns.scatterplot(data = df, x='petal_width', y='petal_length', hue="species")
plt.show()

#%%
# visualizando dados sem categorias
sns.scatterplot(data = df, x='petal_width', y='petal_length')
plt.show()

#%%
# trabalhando os dados
features = df.drop("species", axis=1).values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(features)
normalized_features = scaler.transform(features)

del scaler

#%%
# Aplicando KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(normalized_features)
labels = kmeans.predict(normalized_features)

#%%
# visualizando
sns.scatterplot(data=df, x='petal_width', y='petal_length', hue=labels.astype(str))
plt.show()

#%%
for k in range(1,5):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(features)
    labels = kmeans.predict(features)

    sns.scatterplot(data=df, x='petal_width', y='petal_length', hue=labels.astype(str))
    plt.show()

del k, kmeans

#%%
inertias = []
for k in range(1,11):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)
    
plt.plot(range(1,11), inertias, marker="o")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.show()

del k, kmeans, inertias
