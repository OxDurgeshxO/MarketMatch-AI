# ⚡ MarketMatch-AI: Customer Analytics & Recommendation Engine

An end-to-end Machine Learning pipeline that performs unsupervised retail customer segmentation using **K-Means** and **DBSCAN** algorithms, coupled with an automated **Nearest Neighbors (k-NN)** recommendation engine to power hyper-targeted marketing campaigns.

---

## 🌟 Key Features

- 🎯 **Automated Cluster Optimization**: Evaluates inertia (Elbow Method) and Silhouette Scores to dynamically determine the optimal cluster count $K$.
- 🏷️ **Business Persona Mapping**: Automatically labels customer clusters into actionable business segments (e.g., *High Income - High Spending (Target)*, *Low Income - High Spending (Careless)*).
- 🌌 **Density-Based Outlier Detection**: Leverages **DBSCAN** to group high-density customer clusters and identify anomalous spending profiles.
- 💡 **Profile Similarity Recommendation Engine**: Uses Euclidean Nearest Neighbors to recommend similar customer profiles for lookalike targeting.
- 📊 **Multi-Format Visualizations**: Export high-resolution static PNG charts (2D, 3D, Elbow, Silhouette) and a single-file portable dark-mode **Interactive Plotly HTML Dashboard**.
- 🛡️ **Zero-Config Offline Generator**: Includes an automatic synthetic data fallback generator so the pipeline runs seamlessly even if `Mall_Customers.csv` is not present locally.

---

## 🏗️ Project Architecture

```
MarketMatch-AI/
├── mall_customers_pipeline.py          # Primary entry point & ML pipeline
├── Mall Customers Project              # Legacy single-script file (preserved)
├── Mall_Customers.csv                  # Source Kaggle dataset (Optional/Auto-generated)
├── requirements.txt                    # Project dependency specification
├── README.md                           # Documentation
└── outputs/                            # Exported artifacts
    ├── MarketMatch_Dashboard.html      # Portable Interactive Plotly Dashboard
    ├── Mall_Customers_Segmented_Output.csv
    ├── Mall_Customer_Cluster_Summary.csv
    ├── Elbow_Method.png
    ├── Silhouette_Score.png
    ├── KMeans_Customer_Segments_2D.png
    ├── KMeans_Customer_Segments_3D.png
    └── DBSCAN_Clusters.png
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/OxDurgeshxO/jarvis-ai-assistantcaffine.git
cd MarketMatch-AI-main
pip install -r requirements.txt
```

### 2. Execution

Run the complete pipeline:

```bash
python mall_customers_pipeline.py
```

*Note: If `Mall_Customers.csv` is not present in the workspace, the pipeline automatically generates a synthetic benchmark dataset (`Mall_Customers_Synthetic_Generated.csv`) matching Kaggle statistics.*

---

## 📊 Customer Segmentation Persons

The pipeline categorizes customers based on Annual Income ($k$) and Spending Score (1-100):

| Cluster Persona | Income Range | Spending Score | Marketing Strategy |
|---|---|---|---|
| **High Income - High Spending** | $\ge \$70k$ | $\ge 60$ | Primary target for luxury & VIP loyalty programs |
| **High Income - Low Spending** | $\ge \$70k$ | $\le 40$ | Target with premium upsell offers & incentives |
| **Low Income - High Spending** | $\le \$40k$ | $\ge 60$ | Promotional discounts & trendy trend items |
| **Low Income - Low Spending** | $\le \$40k$ | $\le 40$ | Essential items & value bundles |
| **Middle Income - Moderate** | $\$40k - \$70k$ | $40 - 60$ | Standard general marketing campaigns |

---

## 🌐 Interactive HTML Dashboard

Open `outputs/MarketMatch_Dashboard.html` in any web browser to explore:
1. **Interactive 2D Segments**: Filter and hover over individual customer points.
2. **Interactive 3D Segments**: Rotate and zoom through Age $\times$ Income $\times$ Spending Score dimensions.
3. **DBSCAN Density Distribution**: Inspect cluster boundaries and noise points.

---

## 🔗 Dataset Source

- **Kaggle Dataset**: [Mall Customers Dataset](https://www.kaggle.com/datasets/samanthajones0492/mall-customers-csv)
