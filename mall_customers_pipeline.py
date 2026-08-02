"""
MarketMatch-AI: Retail Customer Segmentation & Recommendation Engine
================================================--------------------
An end-to-end machine learning pipeline for customer segmentation using K-Means
and DBSCAN clustering, paired with a NearestNeighbors similarity recommendation engine.
"""

import os
import sys

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. SETUP & DATA INGESTION (WITH SYNTHETIC FALLBACK)
# -----------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_synthetic_data(num_samples=200, random_state=42):
    """Generate realistic synthetic customer dataset matching Kaggle benchmark structure."""
    np.random.seed(random_state)
    genders = np.random.choice(["Male", "Female"], size=num_samples)
    ages = np.random.randint(18, 70, size=num_samples)
    
    # Create 5 distinct clusters resembling standard Mall Customers distribution
    cluster_centers = [
        (25, 25),   # Low Income - Low Spending
        (25, 80),   # Low Income - High Spending
        (55, 50),   # Middle Income - Moderate Spending
        (85, 20),   # High Income - Low Spending
        (85, 80),   # High Income - High Spending
    ]
    
    incomes = []
    scores = []
    samples_per_cluster = num_samples // len(cluster_centers)
    
    for inc_center, score_center in cluster_centers:
        inc = np.random.normal(inc_center, 8, samples_per_cluster)
        scr = np.random.normal(score_center, 8, samples_per_cluster)
        incomes.extend(inc)
        scores.extend(scr)
        
    incomes = np.clip(np.array(incomes), 15, 140).astype(int)
    scores = np.clip(np.array(scores), 1, 99).astype(int)
    
    df_syn = pd.DataFrame({
        "CustomerID": np.arange(1, num_samples + 1),
        "Genre": genders[:len(incomes)],
        "Age": ages[:len(incomes)],
        "Annual Income (k$)": incomes,
        "Spending Score (1-100)": scores
    })
    return df_syn

def load_data(filepath="Mall_Customers.csv"):
    """Load dataset from disk or automatically fallback to synthetic generator."""
    if os.path.exists(filepath):
        print(f"[OK] Loading dataset from '{filepath}'...")
        return pd.read_csv(filepath)
    else:
        print(f"[WARN] Dataset '{filepath}' not found. Generating synthetic benchmark dataset...")
        df_syn = generate_synthetic_data(num_samples=200)
        df_syn.to_csv("Mall_Customers_Synthetic_Generated.csv", index=False)
        print("[SAVED] Synthetic dataset saved to 'Mall_Customers_Synthetic_Generated.csv'.")
        return df_syn

# -----------------------------------------------------------------------------
# 2. PREPROCESSING & FEATURE SELECTION
# -----------------------------------------------------------------------------
def preprocess(df):
    """Clean and standardize feature matrices for clustering & similarity computation."""
    df_clean = df.copy()
    df_clean = df_clean.rename(columns={
        "Annual Income (k$)": "Income",
        "Spending Score (1-100)": "Score",
        "Genre": "Gender"
    })
    df_clean["Gender_Encoded"] = df_clean["Gender"].map({"Male": 0, "Female": 1})
    
    features_2d = ["Income", "Score"]
    features_3d = ["Age", "Income", "Score"]
    features_rec = ["Age", "Income", "Score", "Gender_Encoded"]
    
    X2 = df_clean[features_2d]
    X3 = df_clean[features_3d]
    X_rec = df_clean[features_rec]
    
    scaler_2d = StandardScaler()
    X2_scaled = scaler_2d.fit_transform(X2)
    
    scaler_3d = StandardScaler()
    X3_scaled = scaler_3d.fit_transform(X3)
    
    scaler_rec = StandardScaler()
    X_rec_scaled = scaler_rec.fit_transform(X_rec)
    
    return df_clean, X2, X3, X2_scaled, X3_scaled, X_rec_scaled, scaler_rec

# -----------------------------------------------------------------------------
# 3. CLUSTERING ENGINE (K-MEANS & DBSCAN)
# -----------------------------------------------------------------------------
def find_optimal_k(X2, k_range=range(2, 11)):
    """Calculate inertia and silhouette scores across cluster ranges."""
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X2)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X2, labels))
        
    best_k = list(k_range)[np.argmax(silhouette_scores)]
    return list(k_range), inertia, silhouette_scores, best_k

def label_segment(income, score):
    """Categorize cluster centroids into intuitive business personas."""
    if income >= 70 and score <= 40:
        return "High Income - Low Spending (Careful)"
    elif income >= 70 and score >= 60:
        return "High Income - High Spending (Target)"
    elif income <= 40 and score >= 60:
        return "Low Income - High Spending (Careless)"
    elif income <= 40 and score <= 40:
        return "Low Income - Low Spending (Sensible)"
    else:
        return "Middle Income - Moderate Spending (Standard)"

def run_kmeans(df, X2, best_k):
    """Fit K-Means model, append labels, and summarize segment metrics."""
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(X2)
    
    cluster_summary = df.groupby("KMeans_Cluster")[["Income", "Score", "Age"]].mean().round(2)
    
    segment_map = {}
    for cluster_id, row in cluster_summary.iterrows():
        segment_map[cluster_id] = label_segment(row["Income"], row["Score"])
        
    df["Segment"] = df["KMeans_Cluster"].map(segment_map)
    return kmeans, cluster_summary, segment_map

def run_dbscan(df, X2_scaled, eps=0.5, min_samples=5):
    """Fit DBSCAN density clustering algorithm."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df["DBSCAN_Cluster"] = dbscan.fit_predict(X2_scaled)
    return dbscan

# -----------------------------------------------------------------------------
# 4. RECOMMENDATION ENGINE
# -----------------------------------------------------------------------------
def build_recommendation_engine(X_rec_scaled):
    """Fit NearestNeighbors model for customer profile similarity lookup."""
    nn_model = NearestNeighbors(n_neighbors=6, metric="euclidean")
    nn_model.fit(X_rec_scaled)
    return nn_model

def recommend_similar_customers(df, X_rec_scaled, nn_model, customer_id, top_n=5):
    """Find top N most similar customer profiles given a CustomerID."""
    customer_row = df[df["CustomerID"] == customer_id]
    if customer_row.empty:
        print(f"[WARN] CustomerID {customer_id} not found.")
        return None
        
    idx = customer_row.index[0]
    distances, indices = nn_model.kneighbors([X_rec_scaled[idx]], n_neighbors=top_n + 1)
    
    rec_indices = indices[0][1:]
    rec_distances = distances[0][1:]
    
    recs = df.loc[rec_indices, ["CustomerID", "Gender", "Age", "Income", "Score", "Segment"]].copy()
    recs["Similarity_Distance"] = rec_distances.round(4)
    return recs

# -----------------------------------------------------------------------------
# 5. STATIC CHART EXPORTS (MATPLOTLIB / SEABORN)
# -----------------------------------------------------------------------------
def export_static_charts(df, kmeans, k_range, inertia, silhouette_scores):
    """Generate high-res PNG plots without blocking execution."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertia, marker='o', color='#1f77b4', linewidth=2)
    ax.set_title("Elbow Method for Optimal K", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Elbow_Method.png"), dpi=150)
    plt.close(fig)
    
    # 2. Silhouette Score
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, silhouette_scores, marker='s', color='#2ca02c', linewidth=2)
    ax.set_title("Silhouette Score for Different K", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Silhouette_Score.png"), dpi=150)
    plt.close(fig)
    
    # 3. K-Means 2D Clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="Income", y="Score", hue="Segment", palette="Set2", s=100, ax=ax
    )
    ax.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        c="black", s=250, marker="X", label="Centroids"
    )
    ax.set_title("K-Means Customer Segments (2D)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "KMeans_Customer_Segments_2D.png"), dpi=150)
    plt.close(fig)
    
    # 4. DBSCAN Clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="Income", y="Score", hue="DBSCAN_Cluster", palette="tab10", s=100, ax=ax
    )
    ax.set_title("DBSCAN Density-Based Clusters", fontsize=14, fontweight='bold')
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "DBSCAN_Clusters.png"), dpi=150)
    plt.close(fig)
    
    # 5. 3D Clusters (Matplotlib)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        df["Age"], df["Income"], df["Score"],
        c=df["KMeans_Cluster"], cmap="viridis", s=60, alpha=0.8
    )
    ax.set_title("3D Customer Clusters (Age x Income x Score)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score")
    fig.colorbar(scatter, label="Cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "KMeans_Customer_Segments_3D.png"), dpi=150)
    plt.close(fig)
    
    print(f"[CHARTS] 5 high-resolution static PNG charts exported to '{OUTPUT_DIR}/'.")

# -----------------------------------------------------------------------------
# 6. INTERACTIVE HTML DASHBOARD GENERATOR (PLOTLY)
# -----------------------------------------------------------------------------
def build_interactive_dashboard(df, k_range, inertia, silhouette_scores):
    """Generate a single-file portable dark-mode Plotly HTML dashboard."""
    fig_2d = px.scatter(
        df, x="Income", y="Score", color="Segment",
        hover_data=["CustomerID", "Age", "Gender"],
        title="Interactive 2D Customer Segments",
        template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    fig_3d = px.scatter_3d(
        df, x="Age", y="Income", z="Score", color="Segment",
        hover_data=["CustomerID", "Gender"],
        title="Interactive 3D Customer Segments",
        template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    fig_dbscan = px.scatter(
        df, x="Income", y="Score", color=df["DBSCAN_Cluster"].astype(str),
        hover_data=["CustomerID", "Age"],
        title="DBSCAN Density Clusters & Outliers",
        template="plotly_dark"
    )
    
    dashboard_path = os.path.join(OUTPUT_DIR, "MarketMatch_Dashboard.html")
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MarketMatch-AI — Customer Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-color: #f8fafc;
            --accent-color: #38bdf8;
        }}
        body {{
            font-family: 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .header h1 {{
            margin: 0;
            color: var(--accent-color);
            font-size: 2.2rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MarketMatch-AI Dashboard</h1>
        <p>Retail Customer Segmentation, Cluster Diagnostics & AI Recommendations</p>
    </div>
    
    <div class="grid">
        <div class="card">
            <div id="plot2d"></div>
        </div>
        <div class="card">
            <div id="plot3d"></div>
        </div>
        <div class="card">
            <div id="plotDbscan"></div>
        </div>
    </div>

    <script>
        const config = {{responsive: true}};
        Plotly.newPlot('plot2d', {fig_2d.to_json()}.data, {fig_2d.to_json()}.layout, config);
        Plotly.newPlot('plot3d', {fig_3d.to_json()}.data, {fig_3d.to_json()}.layout, config);
        Plotly.newPlot('plotDbscan', {fig_dbscan.to_json()}.data, {fig_dbscan.to_json()}.layout, config);
    </script>
</body>
</html>
"""
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"[DASHBOARD] Interactive dashboard exported to '{dashboard_path}'.")

# -----------------------------------------------------------------------------
# 7. MAIN ORCHESTRATION PIPELINE
# -----------------------------------------------------------------------------
def main():
    print("==========================================================")
    print("MarketMatch-AI: Retail Customer Segmentation Pipeline")
    print("==========================================================")
    
    # 1. Load Data
    df = load_data("Mall_Customers.csv")
    
    # 2. Preprocess
    df_clean, X2, X3, X2_scaled, X3_scaled, X_rec_scaled, scaler_rec = preprocess(df)
    
    # 3. Find Optimal K
    k_range, inertia, silhouette_scores, best_k = find_optimal_k(X2)
    print(f"[TARGET] Optimal K-Means clusters calculated via Silhouette Score: K={best_k}")
    
    # 4. Fit K-Means
    kmeans, cluster_summary, segment_map = run_kmeans(df_clean, X2, best_k)
    print("\n[SUMMARY] Cluster Persona Breakdown:")
    print(cluster_summary)
    
    # 5. Fit DBSCAN
    dbscan = run_dbscan(df_clean, X2_scaled)
    print(f"\n[DBSCAN] Clusters Detected: {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)} (Outliers: {list(dbscan.labels_).count(-1)})")
    
    # 6. Fit Recommendation Engine
    nn_model = build_recommendation_engine(X_rec_scaled)
    
    # Sample Recommendation Demonstration
    sample_id = df_clean["CustomerID"].iloc[0]
    recs = recommend_similar_customers(df_clean, X_rec_scaled, nn_model, sample_id, top_n=5)
    print(f"\n[RECS] Nearest Neighbors Recommendations for Customer #{sample_id}:")
    print(recs)
    
    # 7. Save Outputs
    output_segmented_path = os.path.join(OUTPUT_DIR, "Mall_Customers_Segmented_Output.csv")
    output_summary_path = os.path.join(OUTPUT_DIR, "Mall_Customer_Cluster_Summary.csv")
    
    df_clean.to_csv(output_segmented_path, index=False)
    cluster_summary.to_csv(output_summary_path)
    
    print(f"\n[OUTPUTS] Saved CSV Outputs:")
    print(f"   1. {output_segmented_path}")
    print(f"   2. {output_summary_path}")
    
    # 8. Export Visualizations
    export_static_charts(df_clean, kmeans, k_range, inertia, silhouette_scores)
    build_interactive_dashboard(df_clean, k_range, inertia, silhouette_scores)
    
    print("\n[SUCCESS] Pipeline execution complete! All artifacts ready.")

if __name__ == "__main__":
    main()
