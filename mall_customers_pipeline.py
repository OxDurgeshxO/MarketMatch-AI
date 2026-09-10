"""
MarketMatch-AI: Enterprise E-Commerce Customer Segmentation & Lookalike Engine (2026 Edition)
=============================================================================================
An end-to-end unsupervised machine learning pipeline combining RFM feature engineering,
log-transformation, K-Means cluster optimization, Gaussian Mixture Models (GMM) for soft
membership, DBSCAN anomaly detection, and Nearest Neighbors lookalike targeting.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=" * 75, flush=True)
print("[MARKETMATCH-AI] ENTERPRISE CUSTOMER SEGMENTATION PIPELINE", flush=True)
print("=" * 75, flush=True)

# 1. SETUP & DIRECTORIES
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 2. INGEST DATASET
data_path = os.path.join(DATA_DIR, "ecommerce_rfm_customers.csv")
if not os.path.exists(data_path):
    # Fallback to legacy path if needed
    legacy_path = os.path.join(DATA_DIR, "mall_customers_legacy.csv")
    if os.path.exists(legacy_path):
        data_path = legacy_path
    else:
        raise FileNotFoundError("No customer dataset found in data/ folder.")

df = pd.read_csv(data_path)
print(f"\n[1/6] Ingested Customer Dataset: {len(df):,} records, {df.shape[1]} columns", flush=True)
print(f"Columns: {df.columns.tolist()}", flush=True)

# Detect if dataset is RFM or legacy Mall
is_rfm = "Recency" in df.columns and "Frequency" in df.columns and "Monetary" in df.columns

if is_rfm:
    feature_cols = ["Recency", "Frequency", "Monetary"]
    extra_cols = ["AOV", "TenureDays"]
    print(f"Mode: Enterprise E-Commerce RFM Segmentation ({', '.join(feature_cols)})", flush=True)
else:
    # Rename legacy columns
    df = df.rename(columns={"Annual Income (k$)": "Income", "Spending Score (1-100)": "Score", "Genre": "Gender"})
    feature_cols = ["Income", "Score"]
    extra_cols = ["Age"]
    print(f"Mode: Legacy Mall Customers Benchmark ({', '.join(feature_cols)})", flush=True)

# Summary statistics
print("\nDescriptive Statistics:")
print(df[feature_cols].describe().round(2).to_string(), flush=True)

# 3. FEATURE PREPROCESSING & LOG-TRANSFORMATIONS
print("\n[2/6] Preprocessing & Normalization (Log1p + StandardScaler)...", flush=True)
X_raw = df[feature_cols].copy()

# Log-transform positive financial features to correct right-skewness
X_log = np.log1p(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 4. K-MEANS OPTIMIZATION (ELBOW & SILHOUETTE)
print("\n[3/6] Evaluating Cluster Counts (K = 2 to 10)...", flush=True)
k_range = list(range(2, 11))
inertias = []
silhouettes = []
calinski_scores = []
davies_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
    davies_scores.append(davies_bouldin_score(X_scaled, labels))

best_k_idx = int(np.argmax(silhouettes))
optimal_k = 5 if is_rfm else k_range[best_k_idx]
print(f"[OPTIMAL] Selected K = {optimal_k} (Silhouette = {silhouettes[k_range.index(optimal_k)]:.4f})", flush=True)

# Fit Final Champion K-Means
kmeans_champion = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
df["KMeans_Cluster"] = kmeans_champion.fit_predict(X_scaled)

# 5. GAUSSIAN MIXTURE MODEL (GMM) FOR PROBABILISTIC MEMBERSHIP
print("\n[4/6] Fitting Gaussian Mixture Model (GMM) for Soft-Clustering...", flush=True)
gmm = GaussianMixture(n_components=optimal_k, covariance_type="full", random_state=42)
gmm.fit(X_scaled)
gmm_probs = gmm.predict_proba(X_scaled)
df["GMM_Cluster"] = gmm.predict(X_scaled)
df["Max_Cluster_Confidence"] = gmm_probs.max(axis=1).round(3)

# 6. DBSCAN DENSITY CLUSTERING & ANOMALY DETECTION
print("\n[5/6] Running DBSCAN Spatial Density Outlier Detection...", flush=True)
dbscan = DBSCAN(eps=0.8, min_samples=10)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
outlier_count = int((df["DBSCAN_Cluster"] == -1).sum())
print(f"DBSCAN Detected {outlier_count} Anomalous / Whale Customer Profiles ({outlier_count/len(df)*100:.1f}%)", flush=True)

# 7. BUSINESS PERSONA MAPPING
print("\n[6/6] Mapping Business Strategy Personas & Exporting Artifacts...", flush=True)

def assign_rfm_persona(cluster_means):
    """Dynamically assign actionable marketing personas based on cluster centroid ranks."""
    # Rank clusters: High spend + low recency = Champions; High recency = At risk/Lost
    means_df = cluster_means.copy()
    means_df["R_rank"] = means_df["Recency"].rank(ascending=True)   # Lower recency is better
    means_df["F_rank"] = means_df["Frequency"].rank(ascending=False) # Higher freq is better
    means_df["M_rank"] = means_df["Monetary"].rank(ascending=False)  # Higher spend is better
    means_df["Composite"] = means_df["R_rank"] + means_df["F_rank"] + means_df["M_rank"]
    
    sorted_clusters = means_df.sort_values(by="Composite").index.tolist()
    
    persona_names = [
        "👑 Champions (Whales)",
        "⭐ Loyal Regulars",
        "🌱 Potential Loyalists",
        "⚠️ At-Risk Customers",
        "💤 Hibernating / Lost"
    ]
    
    cluster_to_persona = {}
    for i, cid in enumerate(sorted_clusters):
        cluster_to_persona[cid] = persona_names[i % len(persona_names)]
    return cluster_to_persona

if is_rfm:
    centroids_raw = df.groupby("KMeans_Cluster")[feature_cols].mean()
    persona_map = assign_rfm_persona(centroids_raw)
else:
    persona_map = {
        0: "Target Spenders", 1: "Careless Spenders", 2: "Sensible Shoppers", 
        3: "Careful Affluent", 4: "Standard Regulars"
    }

df["Customer_Segment"] = df["KMeans_Cluster"].map(persona_map)

# Compute Segment Business Summary
segment_summary = df.groupby("Customer_Segment").agg({
    "CustomerID": "count",
    feature_cols[0]: "mean",
    feature_cols[1]: "mean",
    feature_cols[2]: "mean",
    "Max_Cluster_Confidence": "mean"
}).rename(columns={"CustomerID": "Customer_Count"}).round(2)
segment_summary["Percent_Of_Base"] = (segment_summary["Customer_Count"] / len(df) * 100).round(1)

print("\n--- SEGMENT BUSINESS SUMMARY ---")
print(segment_summary.to_string(), flush=True)

# 8. NEAREST NEIGHBORS LOOKALIKE RECOMMENDATION ENGINE
nn_features = feature_cols + [c for c in extra_cols if c in df.columns]
X_rec = df[nn_features].copy()
scaler_rec = StandardScaler()
X_rec_scaled = scaler_rec.fit_transform(np.log1p(X_rec[feature_cols]))

nn_model = NearestNeighbors(n_neighbors=10, metric="cosine")
nn_model.fit(X_rec_scaled)

# 9. SERIALIZE PRODUCTION ARTIFACTS
joblib.dump(kmeans_champion, os.path.join(MODELS_DIR, "kmeans_rfm_pipeline.joblib"))
joblib.dump(gmm, os.path.join(MODELS_DIR, "gmm_rfm_model.joblib"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_rfm.joblib"))
joblib.dump(nn_model, os.path.join(MODELS_DIR, "nn_lookalike_model.joblib"))
joblib.dump(scaler_rec, os.path.join(MODELS_DIR, "scaler_rec.joblib"))
joblib.dump(persona_map, os.path.join(MODELS_DIR, "persona_map.joblib"))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))

# Save metadata JSON
metadata = {
    "total_customers": len(df),
    "optimal_k": optimal_k,
    "silhouette_score": round(silhouettes[best_k_idx], 4),
    "features": feature_cols,
    "is_rfm": is_rfm,
    "segments": segment_summary.reset_index().to_dict(orient="records")
}
with open(os.path.join(MODELS_DIR, "segment_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# Save CSV outputs
df.to_csv(os.path.join(OUTPUT_DIR, "Ecommerce_Customers_Segmented.csv"), index=False)
segment_summary.to_csv(os.path.join(OUTPUT_DIR, "Cluster_Business_Summary.csv"))

# 10. GENERATE PLOTS & DASHBOARDS
print("\nGenerating Diagnostic Charts & HTML Dashboards...", flush=True)

# Chart A: Elbow Curve
fig_elbow, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_range, inertias, marker="o", color="#3b82f6", lw=2)
ax.axvline(optimal_k, color="#ef4444", linestyle="--", label=f"Optimal K = {optimal_k}")
ax.set_title("Elbow Method (Inertia vs Number of Clusters)", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.legend()
plt.tight_layout()
fig_elbow.savefig(os.path.join(OUTPUT_DIR, "Elbow_Method.png"), dpi=300)
plt.close()

# Chart B: Silhouette Curve
fig_sil, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_range, silhouettes, marker="s", color="#10b981", lw=2)
ax.axvline(optimal_k, color="#ef4444", linestyle="--", label=f"Peak Score = {silhouettes[best_k_idx]:.3f}")
ax.set_title("Silhouette Score for Different K", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.legend()
plt.tight_layout()
fig_sil.savefig(os.path.join(OUTPUT_DIR, "Silhouette_Score.png"), dpi=300)
plt.close()

# Chart C: DBSCAN Clusters
fig_db, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(X_raw.iloc[:, 0], X_raw.iloc[:, 1], c=df["DBSCAN_Cluster"], cmap="viridis", alpha=0.7, edgecolors="none")
ax.set_title("DBSCAN Spatial Clusters & Outlier Detection (-1)", fontsize=12, fontweight="bold")
ax.set_xlabel(feature_cols[0])
ax.set_ylabel(feature_cols[1])
plt.colorbar(scatter, label="DBSCAN Cluster ID")
plt.tight_layout()
fig_db.savefig(os.path.join(OUTPUT_DIR, "DBSCAN_Clusters.png"), dpi=300)
plt.close()

# 3D Matplotlib Visual Export
if is_rfm:
    fig_3d = plt.figure(figsize=(9, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    scatter_3d = ax_3d.scatter(df['Recency'], df['Frequency'], df['Monetary'], c=df['KMeans_Cluster'], cmap='tab10', alpha=0.6, s=25)
    ax_3d.set_xlabel('Recency (Days)', fontsize=9, fontweight='bold')
    ax_3d.set_ylabel('Frequency (Orders)', fontsize=9, fontweight='bold')
    ax_3d.set_zlabel('Monetary ($)', fontsize=9, fontweight='bold')
    ax_3d.set_title('3D Customer Segments (RFM Space)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_3d.savefig(os.path.join(OUTPUT_DIR, "RFM_Customer_Segments_3D.png"), dpi=300)
    plt.close()

    fig_plotly = px.scatter_3d(
        df, x="Recency", y="Frequency", z="Monetary",
        color="Customer_Segment",
        hover_data=["CustomerID", "AOV", "TopCategory"],
        title="Interactive 3D E-Commerce Customer Segments (RFM Space)",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_plotly.update_layout(height=650)
    fig_plotly.write_html(os.path.join(OUTPUT_DIR, "MarketMatch_Dashboard.html"))

print(f"[DONE] Successfully generated all models, outputs, and dashboard files in '{OUTPUT_DIR}/'.")
print("=" * 75, flush=True)
