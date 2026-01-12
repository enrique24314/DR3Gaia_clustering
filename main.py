import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

#preprocessing
df = pd.read_csv(r"C:\Archivos Carrera\4º curso\Aprendizaje automatico\Bloque 4\gaia_unsupervised\data\dataGaia2.csv")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")
df.info()
nan_frac = df.isna().mean().sort_values(ascending=False)
print(nan_frac.head(15))
df_bad = df[
    (df["Plx"] <= 0) |
    (df["Teff"] <= 0) |
    (df["logg"] < 0) |
    (df["RUWE"] > 1.4)
]

print(f"Filas no físicas: {len(df_bad)}")

df["BP_RP"] = df["BPmag"] - df["RPmag"]

selected_cols = [
    "Teff",
    "logg",
    "[Fe/H]",
    "GMAG",
    "BP_RP",
    "Lum-Flame",
    "Rad-Flame"
]

df_sel = df[selected_cols].copy()

mask = (
    (df["Plx"] > 0) &
    (df["Teff"] > 0) &
    (df["logg"] >= 0) &
    (df["RUWE"] <= 1.4)
)

df_sel = df_sel[mask]
df_sel = df_sel.dropna()
print(df_sel.shape)

df_sel = df_sel.sample(n=200_000, random_state=42)
#PCA
def run_pca(df, n_components=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca, X_scaled
pca, X_pca, X_scaled = run_pca(df_sel)


expl_var = pca.explained_variance_ratio_
cum_var = np.cumsum(expl_var)

for i, v in enumerate(cum_var, start=1):
    print(f"{i}cum_var PCs: {v:.6f}")
for i, v in enumerate(expl_var, start=1):
    print(f"{i}expl_var PCs: {v:.6f}")

#clustering
Z = X_pca[:, :4] 
k_opt = 5
kmeans = KMeans(n_clusters=k_opt, n_init=20, random_state=42)
labels_km = kmeans.fit_predict(Z)

df_clust = df_sel.copy()
df_clust["cluster_km"] = labels_km

#análisis visual

# Visualización de clusters en PC1–PC2
plt.figure(figsize=(6,5))
plt.scatter(
    X_pca[:,0], X_pca[:,1],
    c=labels_km, s=1
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("k-means clusters en PC1–PC2")
plt.tight_layout()
plt.show()

# Visualización de clusters en diagrama HR
plt.figure(figsize=(6,6))
plt.scatter(
    df_clust["BP_RP"],
    df_clust["GMAG"],
    c=df_clust["cluster_km"],
    s=1
)
plt.gca().invert_yaxis()
plt.xlabel("BP − RP")
plt.ylabel("MG")
plt.title("Diagrama HR coloreado por clusters (k=5, PCA)")
plt.tight_layout()
plt.show()

#interpretación numérica
summary = df_clust.groupby("cluster_km")[[
    "Teff", "logg", "[Fe/H]", "GMAG", "Rad-Flame", "Lum-Flame"
]].agg(["mean", "std"])

print(summary)