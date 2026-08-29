import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def generate_customer_data():

    np.random.seed(42)

    n = 1000

    customer_ids = range(1, n + 1)

    age = np.random.randint(
        18,
        70,
        n
    )

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


def perform_clustering(df, n_clusters=5):

    features = [
        "Annual_Income",
        "Spending_Score",
        "Purchase_Frequency"
    ]

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(
        X_scaled
    )

    return df, model


def generate_cluster_summary(df):

    summary = (
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

    return summary


if __name__ == "__main__":

    data = generate_customer_data()

    clustered_data, model = perform_clustering(
        data
    )

    summary = generate_cluster_summary(
        clustered_data
    )

    print("Customer Segmentation Completed Successfully")
    print()

    print("Cluster Summary:")
    print(summary)
