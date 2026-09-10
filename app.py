"""
MarketMatch-AI: Enterprise E-Commerce Customer Analytics & Lookalike Studio (2026 Edition)
==========================================================================================
Interactive Dark-Mode Streamlit Studio featuring 3D Plotly Customer Segmentation,
GMM Probabilistic Soft-Clustering, k-NN Lookalike Audience Generator, and Marketing ROI Simulator.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MarketMatch-AI | Customer Analytics Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 28px;
        border-radius: 16px;
        border-left: 6px solid #38bdf8;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 6px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 8px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .metric-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA INGESTION & CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_rfm_data():
    path = os.path.join("data", "ecommerce_rfm_customers.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_legacy_data():
    path = os.path.join("data", "mall_customers_legacy.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Annual Income (k$)" in df.columns:
            df = df.rename(columns={"Annual Income (k$)": "Income", "Spending Score (1-100)": "Score", "Genre": "Gender"})
        return df
    return None

# Sidebar Dataset Selection
st.sidebar.title("⚡ MarketMatch-AI")
st.sidebar.markdown("---")

dataset_mode = st.sidebar.selectbox(
    "📊 Ingested Customer Dataset",
    ["E-Commerce RFM (4,000 Customers)", "Classic Mall Benchmark (200 Customers)", "Custom CSV Upload"]
)

if dataset_mode == "E-Commerce RFM (4,000 Customers)":
    raw_df = load_rfm_data()
    is_rfm = True
    features = ["Recency", "Frequency", "Monetary"]
elif dataset_mode == "Classic Mall Benchmark (200 Customers)":
    raw_df = load_legacy_data()
    is_rfm = False
    features = ["Income", "Score"]
else:
    uploaded = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded:
        raw_df = pd.read_csv(uploaded)
        is_rfm = "Recency" in raw_df.columns and "Frequency" in raw_df.columns and "Monetary" in raw_df.columns
        features = ["Recency", "Frequency", "Monetary"] if is_rfm else [c for c in raw_df.select_dtypes(include=[np.number]).columns if c != "CustomerID"][:3]
    else:
        raw_df = load_rfm_data()
        is_rfm = True
        features = ["Recency", "Frequency", "Monetary"]

if raw_df is None:
    st.error("Dataset not found. Please ensure data/ directory contains customer records.")
    st.stop()

df = raw_df.copy()

# Sidebar Hyperparameters
st.sidebar.markdown("### ⚙️ Cluster Hyperparameters")
n_clusters = st.sidebar.slider("Number of Segments (K)", min_value=2, max_value=8, value=5)
eps = st.sidebar.slider("DBSCAN Epsilon (Radius)", min_value=0.2, max_value=2.0, value=0.8, step=0.1)
min_samples = st.sidebar.slider("DBSCAN Min Density", min_value=3, max_value=20, value=10)

# -----------------------------------------------------------------------------
# 3. PREPROCESSING & MODELING
# -----------------------------------------------------------------------------
X_raw = df[features].copy()
X_log = np.log1p(X_raw)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["Cluster_ID"] = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, df["Cluster_ID"])

# GMM Probabilistic Clustering
gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
gmm.fit(X_scaled)
gmm_probs = gmm.predict_proba(X_scaled)
df["GMM_Cluster"] = gmm.predict(X_scaled)
df["Confidence"] = gmm_probs.max(axis=1).round(3)

# DBSCAN Anomaly Detection
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
outliers_count = int((df["DBSCAN_Cluster"] == -1).sum())

# Assign Dynamic Business Personas
if is_rfm and n_clusters == 5:
    cluster_means = df.groupby("Cluster_ID")[features].mean()
    cluster_means["R_rank"] = cluster_means["Recency"].rank(ascending=True)
    cluster_means["F_rank"] = cluster_means["Frequency"].rank(ascending=False)
    cluster_means["M_rank"] = cluster_means["Monetary"].rank(ascending=False)
    cluster_means["Composite"] = cluster_means["R_rank"] + cluster_means["F_rank"] + cluster_means["M_rank"]
    sorted_cids = cluster_means.sort_values(by="Composite").index.tolist()
    
    persona_names = [
        "👑 Champions (Whales)",
        "⭐ Loyal Regulars",
        "🌱 Potential Loyalists",
        "⚠️ At-Risk Customers",
        "💤 Hibernating / Lost"
    ]
    persona_map = {cid: persona_names[i] for i, cid in enumerate(sorted_cids)}
    df["Customer_Segment"] = df["Cluster_ID"].map(persona_map)
else:
    df["Customer_Segment"] = df["Cluster_ID"].apply(lambda x: f"Segment #{x + 1}")

# k-NN Lookalike Engine
nn_model = NearestNeighbors(n_neighbors=10, metric="cosine")
nn_model.fit(X_scaled)

# -----------------------------------------------------------------------------
# 4. TOP HERO HEADER & OVERVIEW KPIS
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-box">
    <div style="margin-bottom: 8px;">
        <span class="badge-pill">🚀 ENTERPRISE ANALYTICS</span>
        <span class="badge-pill">🎯 RFM LOG-SCALED</span>
        <span class="badge-pill">🌌 3D PLOTLY ENGINE</span>
        <span class="badge-pill">🤖 GMM SOFT CLUSTERING</span>
    </div>
    <div class="hero-title">⚡ MarketMatch-AI: Customer Segmentation Studio</div>
    <div class="hero-subtitle">Unsupervised Machine Learning & Lookalike Audience Intelligence for Retail Marketers</div>
</div>
""", unsafe_allow_html=True)

# Top 4 KPI Metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{len(df):,}</div>
        <div class="metric-lbl">Total Profiles</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color: #38bdf8;">{n_clusters}</div>
        <div class="metric-lbl">Active Segments</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color: #34d399;">{sil_score:.3f}</div>
        <div class="metric-lbl">Silhouette Quality</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-val" style="color: #f87171;">{outliers_count} ({outliers_count/len(df)*100:.1f}%)</div>
        <div class="metric-lbl">DBSCAN Outliers</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. TABS NAVIGATION INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 2D & 3D Segment Visualizer",
    "🎯 Lookalike Audience Matcher & Exporter",
    "💰 Segment Financials & Marketing ROI",
    "📈 Optimization Diagnostics"
])

# -----------------------------------------------------------------------------
# TAB 1: 2D & 3D SEGMENT VISUALIZER
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Interactive Multi-Dimensional Customer Clusters")
    st.caption("Rotate, zoom, and inspect customer profiles in 3D feature space.")
    
    col_v1, col_v2 = st.columns([1.6, 1])
    
    with col_v1:
        if is_rfm:
            fig3d = px.scatter_3d(
                df, x="Recency", y="Frequency", z="Monetary",
                color="Customer_Segment",
                hover_data=["CustomerID", "AOV"] + (["TopCategory"] if "TopCategory" in df.columns else []),
                title="3D E-Commerce Customer Segments (Recency × Frequency × Monetary)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=520
            )
            fig3d.update_traces(marker=dict(size=4, opacity=0.85))
            st.plotly_chart(fig3d, width="stretch")
        else:
            fig2d = px.scatter(
                df, x="Income", y="Score",
                color="Customer_Segment",
                hover_data=["CustomerID"],
                title="2D Customer Segments (Income × Spending Score)",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Bold,
                height=520
            )
            st.plotly_chart(fig2d, width="stretch")
            
    with col_v2:
        st.markdown("##### 🏷️ Cluster Business Archetypes")
        summary_table = df.groupby("Customer_Segment").agg({
            "CustomerID": "count",
            features[0]: "mean",
            features[1]: "mean",
            features[2] if len(features) > 2 else features[1]: "mean"
        }).round(1).rename(columns={"CustomerID": "Count"})
        summary_table["Share"] = (summary_table["Count"] / len(df) * 100).apply(lambda x: f"{x:.1f}%")
        st.dataframe(summary_table, width="stretch")
        
        st.markdown("##### 💡 Strategic Marketing Recommendations")
        for seg in summary_table.index:
            if "Champions" in seg:
                st.info(f"**{seg}:** VIP Concierge access, early preview of new collections, and referral bonuses.")
            elif "Loyal" in seg:
                st.success(f"**{seg}:** Loyalty rewards program, bundle upsells, and subscription membership perks.")
            elif "Potential" in seg:
                st.warning(f"**{seg}:** Welcome incentives, product onboarding guides, and free shipping promos.")
            elif "At-Risk" in seg:
                st.error(f"**{seg}:** Automated win-back email sequence, limited-time reactivation discount coupon.")
            elif "Hibernating" in seg:
                st.caption(f"**{seg}:** Low-cost re-engagement survey and clearance sale notifications.")

# -----------------------------------------------------------------------------
# TAB 2: LOOKALIKE AUDIENCE MATCHER & EXPORTER
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Customer Lookalike Recommendation & Export Engine")
    st.write("Target lookalike customer cohorts based on multi-dimensional cosine profile similarity.")
    
    col_rec1, col_rec2 = st.columns([1, 1.6])
    
    with col_rec1:
        st.markdown("##### 🔍 Lookalike Profiler")
        selected_id = st.selectbox("Select Benchmark CustomerID:", df["CustomerID"].tolist(), index=0)
        num_recs = st.slider("Number of Lookalikes to Retrieve", min_value=3, max_value=15, value=5)
        
        target_idx = df[df["CustomerID"] == selected_id].index[0]
        distances, indices = nn_model.kneighbors([X_scaled[target_idx]], n_neighbors=num_recs + 1)
        
        # Calculate similarity percentage
        sim_scores = (1 - distances[0][1:]) * 100
        recs_df = df.iloc[indices[0][1:]][["CustomerID", "Customer_Segment"] + features].copy()
        recs_df["Similarity_Match"] = [f"{s:.1f}%" for s in sim_scores]
        
        st.markdown("##### 📁 Export Custom Audience for Ad Platforms")
        export_segment = st.selectbox("Select Segment to Export:", ["All Customers"] + df["Customer_Segment"].unique().tolist())
        
        if export_segment == "All Customers":
            export_df = df
        else:
            export_df = df[df["Customer_Segment"] == export_segment]
            
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download '{export_segment}' Audience CSV ({len(export_df):,} users)",
            data=csv_data,
            file_name=f"marketmatch_audience_{export_segment.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            width="stretch"
        )
        
    with col_rec2:
        target_row = df.iloc[target_idx]
        st.markdown(f"""
        **Benchmark Customer:** `#{target_row['CustomerID']}` | **Segment:** `{target_row['Customer_Segment']}`  
        *Recency:* {target_row['Recency'] if is_rfm else 'N/A'} days | *Frequency:* {target_row['Frequency'] if is_rfm else 'N/A'} orders | *Monetary:* ${target_row['Monetary'] if is_rfm else target_row.get('Income', 'N/A'):,.2f}
        """)
        st.dataframe(recs_df, width="stretch", hide_index=True)
        
        # Radar chart of target vs lookalike mean
        st.markdown("##### Profile Attribute Comparison")
        fig_bar = px.bar(
            recs_df, x="CustomerID", y="Monetary" if is_rfm else features[0],
            color="Customer_Segment", title="Lookalike Spend Comparison ($)",
            template="plotly_dark", height=280
        )
        st.plotly_chart(fig_bar, width="stretch")

# -------------------------------------------------------------
# TAB 3: SEGMENT FINANCIALS & MARKETING ROI
# -------------------------------------------------------------
with tab3:
    st.subheader("Segment Revenue Contribution & Marketing ROI Simulator")
    st.write("Model the economic revenue contribution and projected profit from targeted segment marketing.")
    
    col_fin1, col_fin2 = st.columns([1, 1.2])
    
    with col_fin1:
        if is_rfm:
            fig_rev = px.pie(
                df, names="Customer_Segment", values="Monetary",
                title="Gross Revenue Share by Customer Segment",
                template="plotly_dark", hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_rev, width="stretch")
        else:
            fig_pie = px.pie(df, names="Customer_Segment", title="Customer Distribution", template="plotly_dark")
            st.plotly_chart(fig_pie, width="stretch")
            
    with col_fin2:
        st.markdown("##### 📈 Campaign ROI Simulator")
        target_campaign_seg = st.selectbox("Target Campaign Segment:", df["Customer_Segment"].unique().tolist())
        target_cohort_size = len(df[df["Customer_Segment"] == target_campaign_seg])
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            offer_cost = st.slider("Cost per Outreach / Offer ($)", min_value=5, max_value=150, value=25)
        with col_s2:
            conversion_rate = st.slider("Expected Conversion / Winback (%)", min_value=2, max_value=40, value=15)
            
        avg_spend_seg = df[df["Customer_Segment"] == target_campaign_seg]["Monetary"].mean() if is_rfm else 500.0
        
        campaign_cost = target_cohort_size * offer_cost
        converted_users = int(target_cohort_size * (conversion_rate / 100.0))
        gross_return = converted_users * avg_spend_seg
        net_profit = gross_return - campaign_cost
        roi_ratio = (net_profit / campaign_cost) * 100 if campaign_cost > 0 else 0
        
        st.markdown(f"""
        - **Target Cohort:** {target_cohort_size:,} customers
        - **Total Campaign Budget:** `${campaign_cost:,.0f}`
        - **Converted / Saved Customers:** `{converted_users:,}` customers
        - **Gross Revenue Preserved:** `${gross_return:,.0f}`
        - **Net Incremental Profit:** `<span style="color:#10b981; font-weight:bold;">${net_profit:,.0f}</span>`
        - **Projected Campaign ROI:** `<span style="color:#38bdf8; font-weight:bold;">{roi_ratio:,.0f}%</span>`
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 4: OPTIMIZATION DIAGNOSTICS
# -------------------------------------------------------------
with tab4:
    st.subheader("Clustering Diagnostics & Statistical Validation")
    st.write("Quantitative model validation metrics across K-Means, GMM, and DBSCAN algorithms.")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if os.path.exists(os.path.join("outputs", "Elbow_Method.png")):
            st.image(os.path.join("outputs", "Elbow_Method.png"), caption="Elbow Curve: Inertia vs Number of Clusters", width="stretch")
    with col_d2:
        if os.path.exists(os.path.join("outputs", "Silhouette_Score.png")):
            st.image(os.path.join("outputs", "Silhouette_Score.png"), caption="Silhouette Coefficient Peak Validation", width="stretch")
            
    st.markdown("---")
    col_d3, col_d4 = st.columns(2)
    with col_d3:
        if os.path.exists(os.path.join("outputs", "DBSCAN_Clusters.png")):
            st.image(os.path.join("outputs", "DBSCAN_Clusters.png"), caption="DBSCAN Density Distribution & Outlier Identification (-1)", width="stretch")
    with col_d4:
        st.markdown("##### 🔬 GMM Soft-Clustering Confidence")
        st.caption("Distribution of model confidence across probabilistic assignments.")
        fig_hist = px.histogram(df, x="Confidence", nbins=20, title="GMM Posterior Assignment Confidence Distribution", template="plotly_dark", color_discrete_sequence=["#38bdf8"])
        st.plotly_chart(fig_hist, width="stretch")

st.markdown("---")
st.caption("⚡ MarketMatch-AI | Enterprise E-Commerce Customer Analytics Platform | Powered by Scikit-Learn & Plotly")
