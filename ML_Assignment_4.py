import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def scatter_plot(k):
    np.random.seed(1)

    X_train = np.random.randint(1, 11, 20)
    Y_train = np.random.randint(1, 11, 20)

    labels = []
    for i in range(20):
        if X_train[i] + Y_train[i] <= 11:
            labels.append(0)
        else:
            labels.append(1)

    train_data = np.column_stack((X_train, Y_train))

    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(x, y)
    test_data = np.c_[xx.ravel(), yy.ravel()]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, labels)
    predicted = knn.predict(test_data)

    plt.scatter(test_data[predicted == 0, 0],
                test_data[predicted == 0, 1],
                color='blue', s=5)
    plt.scatter(test_data[predicted == 1, 0],
                test_data[predicted == 1, 1],
                color='red', s=5)

    plt.scatter(X_train, Y_train, c=labels,
                edgecolor='black', s=80)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'kNN (k = {k})')
    plt.show()

def price_predict():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

    df["Label"] = np.where(df["Payment (Rs)"] > 200, 1, 0)

    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Label"].values

    X = np.column_stack((np.ones(len(X)), X))

    W = np.linalg.pinv(X) @ y

    pred = X @ W

    mse = np.mean((y - pred) ** 2)
    rmse = np.sqrt(mse)
    non_zero_idx = y != 0
    if np.any(non_zero_idx):
        mape = np.mean(
            np.abs((y[non_zero_idx] - pred[non_zero_idx]) / y[non_zero_idx])
        ) * 100
    else:
        mape = None

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print("MSE  :", mse)
    print("RMSE :", rmse)

    if mape is not None:
        print("MAPE :", mape)
    else:
        print("MAPE : Not applicable (y contains zeros)")

    print("R2   :", r2)

def project_knn_plot():
    df = pd.read_excel('participants_with_eeg_features.xlsx')

    X = df[['Relative_Alpha', 'Alpha_Theta_Ratio']]
    y = LabelEncoder().fit_transform(df['Group'])

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
    y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = knn.predict(grid)

    plt.scatter(grid[pred == 0, 0], grid[pred == 0, 1],
                color='blue', s=5, alpha=0.3)
    plt.scatter(grid[pred == 1, 0], grid[pred == 1, 1],
                color='red', s=5, alpha=0.3)

    plt.scatter(X.iloc[:, 0], X.iloc[:, 1],
                c=y, edgecolor='black', s=80)

    plt.xlabel('Relative Alpha')
    plt.ylabel('Alpha / Theta Ratio')
    plt.title('Project Data kNN (k = 3)')
    plt.show()

def find_best_k():
    df = pd.read_excel('participants_with_eeg_features.xlsx')

    X = df.drop(columns=['participant_id', 'Group'])
    y = LabelEncoder().fit_transform(df['Group'])

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11]
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy'
    )

    grid.fit(X_scaled, y)

    print("BEST k VALUE:", grid.best_params_['n_neighbors'])
    print("BEST CV ACCURACY:", grid.best_score_)


def main():

    df = pd.read_excel('participants_with_eeg_features.xlsx')

    X = df.drop(columns=['participant_id', 'Group'])
    y = df['Group']

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    y_encoded = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    print("TRAIN CONFUSION MATRIX")
    print(confusion_matrix(y_train, knn.predict(X_train)))
    print(classification_report(y_train, knn.predict(X_train)))

    print("TEST CONFUSION MATRIX")
    print(confusion_matrix(y_test, knn.predict(X_test)))
    print(classification_report(y_test, knn.predict(X_test)))

    price_predict()

    scatter_plot(3)
    for k in range(1, 8):
        scatter_plot(k)

    project_knn_plot()

    find_best_k()

main()
