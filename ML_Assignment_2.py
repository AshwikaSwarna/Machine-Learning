import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
def load_purchase_data(filepath):
    df = pd.read_excel(filepath, sheet_name="Purchase data")
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values.reshape(-1, 1)
    return X, y
def compute_rank(X):
    return np.linalg.matrix_rank(X)
def compute_cost_pseudoinverse(X, y):
    return np.linalg.pinv(X) @ y


def label_customers(y):
    return np.where(y.flatten() > 200, 1, 0)
def train_classifier(X, labels):
    model = LogisticRegression()
    model.fit(X, labels)
    return model


def mean_numpy(data):
    return np.mean(data)
def var_numpy(data):
    return np.var(data)
def mean_manual(data):
    return sum(data) / len(data)
def var_manual(data):
    mu = mean_manual(data)
    return sum((x - mu) ** 2 for x in data) / len(data)
def avg_execution_time(func, data, runs=10):
    times = []
    for _ in range(runs):
        start = time.time()
        func(data)
        times.append(time.time() - start)
    return sum(times) / runs



def jaccard_coefficient(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    denom = f01 + f10 + f11
    return f11 / denom if denom != 0 else 0
def simple_matching_coefficient(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    denom = f00 + f01 + f10 + f11
    return (f11 + f00) / denom if denom != 0 else 0


def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)


def similarity_matrices(binary_data, numeric_data):
    n = len(binary_data)
    jc = np.zeros((n, n))
    smc = np.zeros((n, n))
    cos = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            jc[i, j] = jaccard_coefficient(binary_data[i], binary_data[j])
            smc[i, j] = simple_matching_coefficient(binary_data[i], binary_data[j])
            cos[i, j] = cosine_similarity(numeric_data[i], numeric_data[j])

    return jc, smc, cos


def impute_data(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            if abs(df[col].skew()) < 1:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].median())
    return df


def normalize_data(df):
    scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def main():
    filepath = r"C:\Users\swarn\Downloads\ML Lab.xlsx"

    X, y = load_purchase_data(filepath)
    print("Dimensionality:", X.shape[1])
    print("Number of vectors:", X.shape[0])
    print("Rank of feature matrix:", compute_rank(X))
    print("Product costs:", compute_cost_pseudoinverse(X, y).flatten())

    labels = label_customers(y)
    train_classifier(X, labels)
    print("Classifier trained")

    stock = pd.read_excel(filepath, sheet_name="IRCTC Stock Price")
    stock["Date"] = pd.to_datetime(stock["Date"])
    stock["Day"] = stock["Day"].astype(str).str.strip().str.lower()

    prices = stock.iloc[:, 3].dropna().values

    print("Mean (NumPy):", mean_numpy(prices))
    print("Variance (NumPy):", var_numpy(prices))
    print("Mean (Manual):", mean_manual(prices))
    print("Variance (Manual):", var_manual(prices))

    print("Execution Time NumPy Mean:", avg_execution_time(mean_numpy, prices))
    print("Execution Time Manual Mean:", avg_execution_time(mean_manual, prices))

    wed_prices = stock[stock["Day"] == "wednesday"].iloc[:, 3]
    apr_prices = stock[stock["Date"].dt.month == 4].iloc[:, 3]

    print("Wednesday Mean:", mean_numpy(wed_prices))
    print("April Mean:", mean_numpy(apr_prices))

    print("Probability of Loss:", np.mean(stock["Chg%"] < 0))
    print("P(Profit | Wednesday):", np.mean(stock[stock["Day"] == "wednesday"]["Chg%"] > 0))

    plt.scatter(stock["Day"], stock["Chg%"])
    plt.title("Chg% vs Day")
    plt.show()

    thyroid = pd.read_excel(filepath, sheet_name="thyroid0387_UCI")
    print(thyroid.info())
    print(thyroid.describe())

    binary_cols = [c for c in thyroid.columns if thyroid[c].dropna().isin([0, 1]).all()]
    binary_data = thyroid[binary_cols].head(20).values

    numeric_cols = thyroid.select_dtypes(include=np.number).columns
    numeric_data = MinMaxScaler().fit_transform(thyroid[numeric_cols].head(20).values)

    jc, smc, cos = similarity_matrices(binary_data, numeric_data)

    sns.heatmap(jc); plt.title("Jaccard"); plt.show()
    sns.heatmap(smc); plt.title("SMC"); plt.show()
    sns.heatmap(cos); plt.title("Cosine"); plt.show()

    thyroid = impute_data(thyroid)
    thyroid = normalize_data(thyroid)
    print("Imputation and normalization completed")

if __name__ == "__main__":
    main()

