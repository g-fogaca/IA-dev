#%%
# Aplicando KMeans
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
model.fit(samples)
labels = model.predict(new_samples)

#%%
# Visualizando KMeans
import matplotlib.pyplot as plt
xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs,ys, c=labels)
plt.show()

#%%
import pandas as pd
df = pd.DataFrame({"labels": labels, "species": species})
print(df)

# crosstab
ct = pd.crosstab(df["labels"], df["species"])
print(ct)

# measuring quality (inertia x n_clusters)
print(model.inertia_)

#%%
# testing inertias
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#%%
# normalizando features
from sklearn.preprocessing import StandardScaler
scaler = StandarScaler()
scaler.fit(samples)
samples_scaled = scaler.transform(samples)

#%%
# Fazendo o pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
kmeans = KMeans(n_clusters = 3)

from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)

pipeline.fit(samples)
labels = pipeline.predict(samples)


