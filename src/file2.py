import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# Authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = "rahultomer218"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "3a408d777ea261bafea97717a9c78551fb61a90f"  # ⚠️ use new token

mlflow.set_tracking_uri("https://dagshub.com/rahultomer218/ghughru-dags.mlflow")
mlflow.set_experiment("wine-random-forest-1")

wine = load_wine()
x = wine.data
y = wine.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

n_estimators = 10
max_depth = 50

mlflow.autolog()  # ✅ logs model, params, metrics automatically

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ✅ This autolog does NOT do — must log manually
    mlflow.log_artifact("confusion_matrix.png")

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{cm}")