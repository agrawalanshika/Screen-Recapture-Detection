import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

import joblib

df = pd.read_csv("features.csv")

X = df.drop(["Filename", "Label"], axis=1)

y = df["Label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


param_grid = {

    "n_estimators": [100, 200, 300],

    "max_depth": [None, 5, 10, 20],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4],

    "max_features": ["sqrt", "log2"],

    "bootstrap": [True, False],

    "class_weight": [None, "balanced"]

}


rf = RandomForestClassifier(
    random_state=42
)

grid = GridSearchCV(

    estimator=rf,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=2

)

print("Searching for Best Parameters...\n")

grid.fit(X_train, y_train)


print("BEST PARAMETERS")

print(grid.best_params_)

print("\nBest CV Accuracy :", grid.best_score_)

best_model = grid.best_estimator_

pred = best_model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("FINAL TEST ACCURACY")

print(acc)

print("\nConfusion Matrix")

print(confusion_matrix(y_test, pred))

print("\nClassification Report")

print(classification_report(y_test, pred))

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

classes = ["Real", "Screen"]
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

threshold = cm.max() / 2

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=12,
            fontweight="bold"
        )

plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()


joblib.dump(best_model, "best_random_forest.pkl")

print("\nModel Saved Successfully!")