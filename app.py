"""
MarketMatch-AI — Streamlit Interactive Web Application
=====================================================
A modern, dark-mode web application for retail customer segmentation,
DBSCAN outlier detection, and Nearest Neighbors lookalike recommendations.
"""

import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MarketMatch-AI | Customer Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #38bdf8;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists("Mall_Customers.csv"):
        df = pd.read_csv("Mall_Customers.csv")
    elif os.path.exists("Mall_Customers_Synthetic_Generated.csv"):
        df = pd.read_csv("Mall_Customers_Synthetic_Generated.csv")
    else:
        np.random.seed(42)
        genders = np.random.choice(["Male", "Female"], size=200)
        ages = np.random.randint(18, 70, size=200)
        cluster_centers = [(25, 25), (25, 80), (55, 50), (85, 20), (85, 80)]
        incomes, scores = [], []
        for inc_center, score_center in cluster_centers:
            incomes.extend(np.random.normal(inc_center, 8, 40))
            scores.extend(np.random.normal(score_center, 8, 40))
        df = pd.DataFrame({
            "CustomerID": np.arange(1, 201),
            "Genre": genders,
            "Age": ages,
            "Annual Income (k$)": np.clip(incomes, 15, 140).astype(int),
            "Spending Score (1-100)": np.clip(scores, 1, 99).astype(int)
        })
    
    df_clean = df.rename(columns={
        "Annual Income (k$)": "Income",
        "Spending Score (1-100)": "Score",
        "Genre": "Gender"
    })
    df_clean["Gender_Encoded"] = df_clean["Gender"].map({"Male": 0, "Female": 1})
    return df_clean

df = load_data()

# Features matrices
X2 = df[["Income", "Score"]]
X3 = df[["Age", "Income", "Score"]]
X_rec = df[["Age", "Income", "Score", "Gender_Encoded"]]

scaler_2d = StandardScaler()
X2_scaled = scaler_2d.fit_transform(X2)

scaler_rec = StandardScaler()
X_rec_scaled = scaler_rec.fit_transform(X_rec)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Parameters")
st.sidebar.markdown("---")

n_clusters = st.sidebar.slider("K-Means Clusters (K)", min_value=2, max_value=10, value=5)
eps = st.sidebar.slider("DBSCAN Epsilon (eps)", min_value=0.1, max_value=1.5, value=0.5, step=0.1)
min_samples = st.sidebar.slider("DBSCAN Min Samples", min_value=2, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Dataset Stats**")
st.sidebar.write(f"Total Customers: **{len(df)}**")
st.sidebar.write(f"Avg Income: **${df['Income'].mean():.1f}k**")
st.sidebar.write(f"Avg Spending Score: **{df['Score'].mean():.1f}**")

# -----------------------------------------------------------------------------
# HEADER & OVERVIEW METRICS
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ MarketMatch-AI Web Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Retail Customer Segmentation & Lookalike Recommendation Platform</div>', unsafe_allow_html=True)

# Run K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X2)

def label_segment(income, score):
    if income >= 70 and score <= 40:
        return "High Income - Low Spending"
    elif income >= 70 and score >= 60:
        return "High Income - High Spending (Target)"
    elif income <= 40 and score >= 60:
        return "Low Income - High Spending"
    elif income <= 40 and score <= 40:
        return "Low Income - Low Spending"
    else:
        return "Middle Income - Moderate"

cluster_summary = df.groupby("KMeans_Cluster")[["Income", "Score", "Age"]].mean()
segment_map = {cid: label_segment(row["Income"], row["Score"]) for cid, row in cluster_summary.iterrows()}
df["Segment"] = df["KMeans_Cluster"].map(segment_map)

sil_score = silhouette_score(X2, df["KMeans_Cluster"])

# Run DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X2_scaled)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Optimal Clusters (K)", n_clusters)
with col2:
    st.metric("Silhouette Score", f"{sil_score:.3f}")
with col3:
    st.metric("DBSCAN Clusters", len(set(df["DBSCAN_Cluster"])) - (1 if -1 in df["DBSCAN_Cluster"] else 0))
with col4:
    st.metric("DBSCAN Outliers", list(df["DBSCAN_Cluster"]).count(-1))

st.markdown("---")

# -----------------------------------------------------------------------------
# TABS INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 2D & 3D Customer Segments",
    "🔍 DBSCAN Outlier Detection",
    "💡 Similarity Recommendations",
    "📈 Optimization Diagnostics"
])

# Tab 1: 2D & 3D Plots
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("2D Customer Segments (Income × Score)")
        fig2d = px.scatter(
            df, x="Income", y="Score", color="Segment",
            hover_data=["CustomerID", "Age", "Gender"],
            template="plotly_dark", height=450,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig2d, use_container_width=True)
    
    with col_b:
        st.subheader("3D Customer Segments (Age × Income × Score)")
        fig3d = px.scatter_3d(
            df, x="Age", y="Income", z="Score", color="Segment",
            hover_data=["CustomerID", "Gender"],
            template="plotly_dark", height=450,
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig3d, use_container_width=True)

# Tab 2: DBSCAN
with tab2:
    st.subheader("DBSCAN Density Clustering & Outlier Detection")
    fig_db = px.scatter(
        df, x="Income", y="Score", color=df["DBSCAN_Cluster"].astype(str),
        hover_data=["CustomerID", "Age", "Gender"],
        labels={"color": "DBSCAN Cluster (-1 = Outlier)"},
        template="plotly_dark", height=500
    )
    st.plotly_chart(fig_db, use_container_width=True)

# Tab 3: Recommendation Engine
with tab3:
    st.subheader("Customer Lookalike Recommendation Engine")
    nn_model = NearestNeighbors(n_neighbors=6, metric="euclidean")
    nn_model.fit(X_rec_scaled)
    
    selected_id = st.selectbox("Select CustomerID to find lookalikes:", df["CustomerID"].tolist(), index=0)
    top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
    
    if st.button("Generate Recommendations"):
        idx = df[df["CustomerID"] == selected_id].index[0]
        distances, indices = nn_model.kneighbors([X_rec_scaled[idx]], n_neighbors=top_n + 1)
        
        rec_df = df.loc[indices[0][1:], ["CustomerID", "Gender", "Age", "Income", "Score", "Segment"]].copy()
        rec_df["Similarity_Distance"] = distances[0][1:].round(4)
        
        target_cust = df[df["CustomerID"] == selected_id].iloc[0]
        st.markdown(f"**Target Customer:** #{target_cust['CustomerID']} ({target_cust['Gender']}, Age {target_cust['Age']}, Income ${target_cust['Income']}k, Score {target_cust['Score']})")
        st.dataframe(rec_df, use_container_width=True)

# Tab 4: Optimization Diagnostics
with tab4:
    st.subheader("K-Means Cluster Diagnostics (Elbow & Silhouette Curve)")
    
    k_range = range(2, 11)
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X2)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X2, lbls))
        
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_el = px.line(x=list(k_range), y=inertias, markers=True, title="Elbow Curve (Inertia)", labels={"x": "K", "y": "Inertia"}, template="plotly_dark")
        st.plotly_chart(fig_el, use_container_width=True)
    with col_d2:
        fig_sil = px.line(x=list(k_range), y=sils, markers=True, title="Silhouette Scores", labels={"x": "K", "y": "Score"}, template="plotly_dark")
        st.plotly_chart(fig_sil, use_container_width=True)

st.markdown("---")
st.caption("⚡ Powered by MarketMatch-AI | scikit-learn & Plotly & Streamlit")
