# %% [CELDA 1] IMPORTACIÓN Y CONFIGURACIÓN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Preprocesamiento
from sklearn.preprocessing import StandardScaler

# Reducción de Dimensionalidad
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE

# Clustering
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors  # Para calcular eps óptimo

# Métricas
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, confusion_matrix

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk", font_scale=0.9)


path_training = r'C:\Universidad\Machine Learning\proyectoML\data\UJIndoorLoc\trainingData.csv'
path_validation = r'C:\Universidad\Machine Learning\proyectoML\data\UJIndoorLoc\validationData.csv'

df_train = pd.read_csv(path_training)
try:
    df_val = pd.read_csv(path_validation)
    df = pd.concat([df_train, df_val], ignore_index=True)

except:
    df = df_train
    print(f"   Solo Training cargado. Filas: {len(df)}")

X_raw = df.iloc[:, 0:520]
y_true = df['BUILDINGID']

# Limpieza (+100 -> -105)
print("2. Corrigiendo señal (+100 a -105)")
X = X_raw.replace(100, -105)

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% COMPARATIVA DE REDUCCIÓN DE DIMENSIONALIDAD (TIEMPO vs VARIANZA)

print("Reduccion de la dimensionalidad")
# Vamos a comparar 3 métodos para bajar a 50 dimensiones

methods = [
    ("PCA", PCA(n_components=50)),
    ("SVD (Truncated)", TruncatedSVD(n_components=50)),
    ("Random Projection", GaussianRandomProjection(n_components=50, random_state=42))
]

results_dim = []
reduced_data = {}

for name, model in methods:
    start_time = time.time()
    X_red = model.fit_transform(X_scaled)
    end_time = time.time()

    elapsed = end_time - start_time
    reduced_data[name] = X_red

    results_dim.append({
        "Método": name,
        "Tiempo (s)": elapsed,
        "Dims Salida": 50
    })
    print(f"   {name} completado en {elapsed:.4f}s")


df_dim_results = pd.DataFrame(results_dim)
plt.figure(figsize=(8, 5))
sns.barplot(x="Método", y="Tiempo (s)", data=df_dim_results, palette="magma", hue="Método", legend=False)
plt.title("Eficiencia Computacional: Reducción de Dimensionalidad")
plt.ylabel("Tiempo (segundos)")
plt.show()

X_pca_50 = reduced_data["PCA"]


print("Generando mapa t-SNE 2D para visualizar resultados")
tsne = TSNE(n_components=2, perplexity=40, init='pca', learning_rate='auto', n_jobs=-1)
X_viz = tsne.fit_transform(X_scaled)

# %%  OPTIMIZACIÓN DE HIPERPARÁMETROS

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# 1. ELBOW METHOD para K-Means
print("Calculando Elbow Method para K-Means")
inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=3)
    km.fit(X_pca_50)
    inertias.append(km.inertia_)

ax[0].plot(K_range, inertias, 'bo-', markersize=8)
ax[0].set_title('Elbow Method (Optimización K)')
ax[0].set_xlabel('Número de Clusters (K)')
ax[0].set_ylabel('Distancia intra-cluster')
ax[0].axvline(x=3, color='r', linestyle='--', label='K=3')
ax[0].legend()

# 2. K-DISTANCE GRAPH para DBSCAN
print("Calculando K-Distance para DBSCAN")
# Calculamos la distancia al vecino número 20 (min_samples)
neigh = NearestNeighbors(n_neighbors=20)
nbrs = neigh.fit(X_pca_50)
distances, indices = nbrs.kneighbors(X_pca_50)
distances = np.sort(distances[:, 19], axis=0)  # Distancia al 20vo vecino

ax[1].plot(distances)
ax[1].set_title('K-Distance (Optimización Eps)')
ax[1].set_ylabel('Distancia k-NN')
ax[1].set_xlabel('Puntos ordenados por distancia')
# El "codo" de esta curva es el epsilon óptimo.
# Visualmente suele estar donde la curva sube disparada.
ax[1].set_ylim(0, 15)  # Zoom para ver el codo
ax[1].axhline(y=7, color='r', linestyle='--', label='Eps ~ 7')
ax[1].legend()

plt.show()

# %%  EJECUCIÓN Y COMPARACIÓN DE MODELOS

print("--- ENTRENAMIENTO Y EVALUACIÓN (FULL DATA) ---")

models_to_test = [
    ("K-Means", KMeans(n_clusters=3, random_state=42, n_init=10), X_pca_50),
    ("Mean Shift", MeanShift(bin_seeding=True, bandwidth=estimate_bandwidth(X_pca_50, quantile=0.2, n_samples=500)),
     X_pca_50),
    ("DBSCAN", DBSCAN(eps=7, min_samples=20), X_pca_50),
    ("GMM", GaussianMixture(n_components=3, random_state=42), X_pca_50)
]

results_model = []
labels_dict = {}

for name, model, data in models_to_test:
    print(f"Entrenando {name}...")
    start = time.time()

    # Entrenar
    labels = model.fit_predict(data)

    end = time.time()

    # Guardar etiquetas
    labels_dict[name] = labels

    # Métricas
    ari = adjusted_rand_score(y_true, labels)

    # --- CAMBIO AQUÍ: SILHOUETTE COMPLETO ---
    if len(set(labels)) > 1:
        print(f"   Calculando Silhouette para {name} (usando todos los datos)")
        # Al borrar 'sample_size', usa los 20,000 datos exactos.
        sil = silhouette_score(X_pca_50, labels)
    else:
        sil = -1

    results_model.append({
        "Algoritmo": name,
        "Tiempo (s)": end - start,
        "ARI (Calidad Real)": ari,
        "Silhouette (Cohesión)": sil,
        "Clusters": len(set(labels)) - (1 if -1 in labels else 0)
    })

df_results = pd.DataFrame(results_model)
print("\TABLA FINAL (Métricas Exactas) ")
print(df_results)

# %% VISUALIZACIÓN DE RESULTADOS

# 1. Gráficos de dispersión (Los Mapas)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (name, labels) in enumerate(labels_dict.items()):
    # Usamos t-SNE como mapa base
    # Si DBSCAN tiene ruido, lo pintamos gris
    palette = 'Set1'
    if name == "DBSCAN": palette = "Spectral"

    sns.scatterplot(x=X_viz[:, 0], y=X_viz[:, 1], hue=labels, palette=palette, s=10, ax=axes[i], legend=False)
    axes[i].set_title(f"{name} (ARI: {df_results.loc[i, 'ARI (Calidad Real)']:.2f})")

plt.tight_layout()
plt.show()

# 2. Comparativa de Métricas (Barras)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x="Algoritmo", y="ARI (Calidad Real)", data=df_results, ax=axes[0], palette="viridis", hue="Algoritmo",
            legend=False)
axes[0].set_title("Precisión vs Realidad (ARI)")
axes[0].set_ylim(0, 1)

sns.barplot(x="Algoritmo", y="Silhouette (Cohesión)", data=df_results, ax=axes[1], palette="magma", hue="Algoritmo",
            legend=False)
axes[1].set_title("Separación de Clusters (Silhouette)")

sns.barplot(x="Algoritmo", y="Tiempo (s)", data=df_results, ax=axes[2], palette="Reds", hue="Algoritmo", legend=False)
axes[2].set_title("Tiempo de Ejecución")

plt.show()

# %%  MATRIZ DE CONFUSIÓN (K-MEANS)
# Como los clusters son arbitrarios (0,1,2), a veces el cluster 0 es el edificio 2.


best_labels = labels_dict["K-Means"]  # O el que haya salido mejor
conf_mat = pd.crosstab(y_true, best_labels, rownames=['Edificio REAL'], colnames=['Cluster PREDICHO'])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Matriz de Confusión: K-Means vs Realidad")
plt.show()

print("CONCLUSIÓN FINAL:")
best_model = df_results.loc[df_results['ARI (Calidad Real)'].idxmax()]
print(f"El mejor modelo fue {best_model['Algoritmo']} con un ARI de {best_model['ARI (Calidad Real)']:.3f}.")