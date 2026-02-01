import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score,accuracy_score
from sklearn.preprocessing import LabelEncoder
import math


def dot(A, B):
    if len(A) != len(B):
        print("Vectors should be of same size")
    dot = 0
    for i in range(len(A)):
        dot += A[i] * B[i]
    return dot


def euclidean(x):
    s = 0
    for i in range(len(x)):
        s += x[i] ** 2
    return math.sqrt(s)


def mean(data):
    return sum(data) / len(data)


def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def std(data):
    return math.sqrt(variance(data))


def matrix(m):
    means = []
    stds = []
    for i in range(m.shape[1]):
        col = m[:, i]
        means.append(mean(col))
        stds.append(std(col))
    return np.array(means), np.array(stds)


def density_pattern(x):
    df = pd.read_excel("datasetEEG.xlsx")
    feature3 = df[x].dropna()
    mean3 = np.mean(feature3)
    var3 = np.var(feature3)
    print(f"Mean of {x}:", mean3)
    print(f"Variance of {x}:", var3)

    plt.figure()
    plt.hist(feature3, bins=10)
    plt.title(f"Histogram of {x} Feature")
    plt.xlabel(f"{x} Values")
    plt.ylabel("Frequency")
    plt.show()


def minkowskidist(x, y, p):
    dist = 0
    for i in range(len(x)):
        dist += abs(x[i] - y[i]) ** p
    return dist ** (1/p)


def knnfun(X_train, y_train, X_test, k=3, p=2):
    predictions = []
    for t in X_test:
        distances = []
        for i in range(len(X_train)):
            d = minkowskidist(t, X_train[i], p)
            distances.append((d, y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        labels = [label for _, label in neighbors]
        prediction = max(set(labels), key=labels.count)
        predictions.append(prediction)
    return np.array(predictions)


def confusionmat(yt, yp):
    labels = np.unique(yt)
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    label_index = {label: i for i, label in enumerate(labels)}
    for i in range(len(yt)):
        t = label_index[yt[i]]
        p = label_index[yp[i]]
        cm[t][p] += 1
    return cm, labels


def accuracy(cm):
    return np.trace(cm) / np.sum(cm)

def precision(cm):
    n = cm.shape[0]
    precisions = []

    for i in range(n):
        TP = cm[i][i]
        FP = np.sum(cm[:, i]) - TP

        if TP + FP == 0:
            precisions.append(0)
        else:
            precisions.append(TP / (TP + FP))

    return np.mean(precisions)

def recall(cm):
    n = cm.shape[0]
    recall = []

    for i in range(n):
        TP = cm[i][i]
        FN = np.sum(cm[i, :]) - TP

        if TP + FN == 0:
            recall.append(0)
        else:
            recall.append(TP / (TP + FN))

    return np.mean(recall)

def f1(cm):
    n = cm.shape[0]
    f = []

    for i in range(n):
        TP = cm[i][i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f.append(f1)

    return np.mean(f1)




def main():

    A = np.array([1, 2, 3])
    B = np.array([5, 6, 7])

    print("Dot Product :", dot(A, B))
    print("Dot Product(using numpy) :", np.dot(A, B))

    print("Euclidean Norm  of A :", euclidean(A))
    print("Euclidean Norm (using numpy) of A :", np.linalg.norm(A))

    print("Euclidean Norm of B :", euclidean(B))
    print("Euclidean Norm of B (using numpy) :", np.linalg.norm(B))


    df = pd.read_excel("datasetEEG.xlsx")

    classes = df["Mental_State"].unique()
    class1 = df[df["Mental_State"] == classes[0]]
    class2 = df[df["Mental_State"] == classes[1]]

    features = [col for col in df.columns if col.startswith("EEG")]

    X1 = class1[features].values
    X2 = class2[features].values

    c1, s1 = matrix(X1)
    c2, s2 = matrix(X2)

    print("Class 1 Centroid :", c1)
    print("Class 2 Centroid :", c2)
    print("Class 1 Spread :", s1)
    print("Class 2 Spread :", s2)

    print("Interclass Distance between centroids:", np.linalg.norm(c1 - c2))


    density_pattern("EEG2")


    f14 = df.loc[0, features].values
    f24 = df.loc[1, features].values

    distances = []

    for p in range(1, 11):
        distances.append(minkowskidist(f14, f24, p))

    for i in range(1, 11):
        print(f"minkowski distance {i} : {distances[i-1]}")

    plt.figure()
    plt.plot(range(1, 11), distances)
    plt.xlabel("p value")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance vs p")
    plt.show()

    for p in range(1, 11):
        scipy_dist = minkowski(f14, f24, p)
        print(f"minkowski distance (using scipy) {p} : {scipy_dist}")


    X6 = df[features].values
    Y6 = df["Mental_State"].values


    X_train, X_test, y_train, y_test = train_test_split(
        X6, Y6, test_size=0.2, stratify=Y6, random_state=42
    )

    print("Total samples:", len(X6))
    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)
    print("Training labels size:", y_train.shape)
    print("Testing labels size:", y_test.shape)


    n = KNeighborsClassifier(n_neighbors=3)
    n.fit(X_train, y_train)

    y = n.predict(X_test)
    print("Prediction :", y)
    print("Actual labels:", y_test)

    acc = n.score(X_test, y_test)
    print("Accuracy (built in function) :", acc)


    y_pred = knnfun(X_train, y_train, X_test, k=3)
    accown = np.mean(y_pred == y_test)
    print("Accuracy (own function):", accown)


    accuracies = []
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acck = knn.score(X_test, y_test)
        accuracies.append(acck)
        print(f"k = {k}, Accuracy = {acck}")

    print("Accuracy for k = 1 : ", accuracies[0])
    print("Accuracy for k = 3 : ", accuracies[2])

    plt.figure()
    plt.plot(range(1, 11), accuracies, marker='o')
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k for kNN Classifier")
    plt.grid(True)
    plt.show()


    ytrain = n.predict(X_train)
    ytest = n.predict(X_test)

    print("Confusion Matrix (Training Data):")
    print(confusion_matrix(y_train, ytrain))

    print("Confusion Matrix (Testing Data):")
    print(confusion_matrix(y_test, ytest))

    print("Training :")
    print("Precision:", precision_score(y_train, ytrain, average='weighted'))
    print("Recall   :", recall_score(y_train, ytrain, average='weighted'))
    print("F1-score :", f1_score(y_train, ytrain, average='weighted'))

    print("Testing :")
    print("Precision:", precision_score(y_test, ytest, average='weighted'))
    print("Recall   :", recall_score(y_test, ytest, average='weighted'))
    print("F1-score :", f1_score(y_test, ytest, average='weighted'))

    conmat, labels = confusionmat(y_test, y_pred)

    print("Confusion Matrix:")
    print(conmat)
    print("Labels:", labels)

    prec = precision(conmat)
    rec = recall(conmat)
    f1score = f1(conmat)

    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1score)


    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    Y_train_oh = np.eye(len(np.unique(y_train_enc)))[y_train_enc]

    W = np.linalg.pinv(X_train) @ Y_train_oh

    scores = X_test @ W
    y_pred_mat = np.argmax(scores, axis=1)
    y_pred_mat = le.inverse_transform(y_pred_mat)

    acc_mat = accuracy_score(y_test, y_pred_mat)

    print("Matrix Inversion Accuracy:", acc_mat)


main()
