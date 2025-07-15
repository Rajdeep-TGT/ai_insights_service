from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("app/data/mock_receipts.csv")
    return df

def compute_features(df):
    df["vendor_code"] = df["vendor"].astype("category").cat.codes
    df["category_code"] = df["category"].astype("category").cat.codes
    return df[["amount", "vendor_code", "category_code"]]

def analyze_receipt(receipt):
    df = load_data()
    X = compute_features(df)

    # Train kNN model
    model = NearestNeighbors(n_neighbors=3)
    model.fit(X)

    # Convert receipt input to vector
    vendor_list = df["vendor"].astype("category").cat.categories.tolist()
    category_list = df["category"].astype("category").cat.categories.tolist()

    new_point = np.array([[receipt.amount,
                           vendor_list.index(receipt.vendor) if receipt.vendor in vendor_list else -1,
                           category_list.index(receipt.category) if receipt.category in category_list else -1]])

    distances, _ = model.kneighbors(new_point)
    score = float(np.mean(distances))

    # Duplicate logic
    is_duplicate = bool(df.apply(lambda row: row["amount"] == receipt.amount and 
                             row["vendor"] == receipt.vendor and 
                             row["category"] == receipt.category, axis=1).any())


    tags = []
    if score > 0.8: tags.append("suspicious")
    if is_duplicate: tags.append("duplicate")

    return {
        "is_duplicate": is_duplicate,
        "anomaly_score": round(score, 2),
        "tags": tags
    }
