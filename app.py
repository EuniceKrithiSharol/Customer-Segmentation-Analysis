import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="👥",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("👥 Customer Segmentation Analysis")
st.markdown(
    "Machine Learning based customer segmentation using "
    "K-Means clustering and behavioral analytics."
)

# -------------------------------------------------
# GENERATE DATA
# -------------------------------------------------

@st.cache_data
def generate_customer_data():

    np.random.seed(42)

    n = 1000

    customer_ids = range(1, n + 1)

    age = np.random.randint(18, 70, n)

    annual_income = np.random.randint(
        15000,
        150000,
        n
    )

    spending_score = np.random.randint(
        1,
        101,
        n
    )

    purchase_frequency = np.random.randint(
        1,
        50,
        n
    )

    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Age": age,
        "Annual_Income": annual_income,
        "Spending_Score": spending_score,
        "Purchase_Frequency": purchase_frequency
    })

    return df


df = generate_customer_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("⚙️ Clustering Settings")

num_clusters = st.sidebar.slider(
    "Number of Customer Segments",
    min_value=2,
    max_value=8,
    value=5
)

# -------------------------------------------------
# MACHINE LEARNING
# -------------------------------------------------

features = [
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency"
]

X = df[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=num_clusters,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👥 Total Customers",
    f"{len(df):,}"
)

col2.metric(
    "🎯 Customer Segments",
    num_clusters
)

col3.metric(
    "💰 Average Income",
    f"${df['Annual_Income'].mean():,.0f}"
)

col4.metric(
    "🛒 Average Spending Score",
    f"{df['Spending_Score'].mean():.1f}"
)

st.divider()

# -------------------------------------------------
# CLUSTER VISUALIZATION
# -------------------------------------------------

st.subheader("📊 Customer Segments")

fig_clusters = px.scatter(
    df,
    x="Annual_Income",
    y="Spending_Score",
    color="Cluster",
    size="Purchase_Frequency",
    hover_data=[
        "Customer_ID",
        "Age"
    ],
    title="Customer Segmentation using K-Means Clustering"
)

st.plotly_chart(
    fig_clusters,
    use_container_width=True
)

# -------------------------------------------------
# SEGMENT ANALYSIS
# -------------------------------------------------

st.subheader("📈 Customer Segment Analysis")

cluster_summary = (
    df
    .groupby("Cluster")
    .agg(
        Average_Age=("Age", "mean"),
        Average_Income=(
            "Annual_Income",
            "mean"
        ),
        Average_Spending=(
            "Spending_Score",
            "mean"
        ),
        Average_Purchase_Frequency=(
            "Purchase_Frequency",
            "mean"
        ),
        Customer_Count=(
            "Customer_ID",
            "count"
        )
    )
    .reset_index()
)

st.dataframe(
    cluster_summary,
    use_container_width=True
)

# -------------------------------------------------
# CLUSTER DISTRIBUTION
# -------------------------------------------------

fig_distribution = px.bar(
    cluster_summary,
    x="Cluster",
    y="Customer_Count",
    title="Number of Customers in Each Segment"
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True
)

# -------------------------------------------------
# CUSTOMER DISTRIBUTION
# -------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    fig_income = px.histogram(
        df,
        x="Annual_Income",
        color="Cluster",
        nbins=30,
        title="Income Distribution by Segment"
    )

    st.plotly_chart(
        fig_income,
        use_container_width=True
    )


with col2:

    fig_spending = px.box(
        df,
        x="Cluster",
        y="Spending_Score",
        title="Spending Score by Segment"
    )

    st.plotly_chart(
        fig_spending,
        use_container_width=True
    )

# -------------------------------------------------
# CUSTOMER DATA
# -------------------------------------------------

st.subheader("📄 Customer Dataset")

st.dataframe(
    df,
    use_container_width=True
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Customer Segmentation Analysis | "
    "Python • Scikit-learn • K-Means • Streamlit"
)
