import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

RAW_EEG_FOLDER = r"C:\Users\swarn\ML\EEG_Raw_CSV"
PARTICIPANTS_FILE = r"C:\Users\swarn\ML\participants.tsv"


print("Building feature matrix...")

participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t")
participants = participants.sort_values("participant_id")

feature_rows = []

for file in sorted(os.listdir(RAW_EEG_FOLDER)):

    if not file.endswith(".csv"):
        continue

    subject_id = file.replace("_rawEEG.csv", "")
    file_path = os.path.join(RAW_EEG_FOLDER, file)

    df = pd.read_csv(file_path, nrows=60000)

    subject_features = {"participant_id": subject_id}

    means = df.mean()

    for col in df.columns:
        subject_features[col] = means[col]

    feature_rows.append(subject_features)

feature_df = pd.DataFrame(feature_rows)

target_column = "MMSE"   

dataset = feature_df.merge(
    participants[["participant_id", target_column]],
    on="participant_id"
)

dataset.rename(columns={target_column: "target"}, inplace=True)

print("Dataset shape:", dataset.shape)

X = dataset.drop(columns=["participant_id", "target"]).values
y = dataset["target"].values

print("Linear Regression ")

X_single = X[:, 0].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

print("Evaluation Metrics")

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

print("Train Metrics:", evaluate(y_train, y_train_pred))
print("Test  Metrics:", evaluate(y_test, y_test_pred))

print("Linear Regression (All Features)")

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg_all = LinearRegression().fit(X_train_all, y_train_all)

y_train_pred_all = reg_all.predict(X_train_all)
y_test_pred_all = reg_all.predict(X_test_all)

print("Train Metrics:", evaluate(y_train_all, y_train_pred_all))
print("Test  Metrics:", evaluate(y_test_all, y_test_pred_all))

print("K-Means Clustering (k=2)")

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X)
labels = kmeans.labels_

print("Cluster Centers:", kmeans.cluster_centers_)

print("Clustering Scores")

print("Silhouette Score:", silhouette_score(X, labels))
print("Calinski-Harabasz Score:", calinski_harabasz_score(X, labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X, labels))

print("Evaluating different k values")

k_values = range(2, 10)

sil_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    labels_k = km.labels_

    sil_scores.append(silhouette_score(X, labels_k))
    ch_scores.append(calinski_harabasz_score(X, labels_k))
    db_scores.append(davies_bouldin_score(X, labels_k))

plt.figure()
plt.plot(k_values, sil_scores, label="Silhouette")
plt.plot(k_values, ch_scores, label="CH Score")
plt.plot(k_values, db_scores, label="DB Index")
plt.xlabel("k")
plt.title("Clustering Evaluation Scores")
plt.legend()
plt.show()

print("Elbow Plot")

distortions = []

for k in range(2, 20):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    distortions.append(km.inertia_)

plt.figure()
plt.plot(range(2, 20), distortions)
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.show()