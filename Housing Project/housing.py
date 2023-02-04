# importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importando dataset
<<<<<<< HEAD
df = pd.read_csv(r"C:\\Users\\gabfo\\OneDrive\\Documentos\\Programação\\FEA dev\\IA dev\Housing Project\housing.csv")
=======
df = pd.read_csv("housing.csv")
>>>>>>> e1a632394b5a4d3aea79acf3fadab97f8fa82a82
df.info()
features = df.values

#%%
# funções
def plot(df,x,y,hue=None):
    sns.set_style("whitegrid")
    plt.figure(figsize=(5,5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.show()
    
def boxplot(df,x,y):
    sns.boxplot(data=df, x=x, y=y, showfliers=False)
    plt.show()

# visualizando
plot(df,"Longitude", "Latitude")

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
    kmeans = KMeans(n_clusters = k, random_state=(0))
    kmeans.fit(features_1)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker="o")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.show()

del inertias, k, kmeans

#%%
for k in range(1,7):
    kmeans = KMeans(n_clusters = k, random_state=(0))
    kmeans.fit(features_1)
    labels = kmeans.predict(features_1)
    
    df['Region'] = labels.astype(str)

    plot(df,"Longitude", "Latitude", "Region")

del k, kmeans, labels, features

#%%
<<<<<<< HEAD
for k in range(1,7):
    kmeans = KMeans(n_clusters = k, random_state=(0))
    kmeans.fit(features_1)
    labels = kmeans.predict(features_1)
    
    df['Region'] = labels.astype(str)

    boxplot(df, "Region", "MedHouseVal")
=======
df.boxplot("MedHouseVal", "Region")
plt.show()
>>>>>>> e1a632394b5a4d3aea79acf3fadab97f8fa82a82
