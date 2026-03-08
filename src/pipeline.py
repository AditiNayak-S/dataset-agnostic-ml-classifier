import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# --- 1. Data Loading ---
def load_dataset(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

# --- 2. Preprocessing ---
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

# --- 3. Model Comparison ---
def compare_models(preprocessor, X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

    trained_models = {}
    leaderboard = []

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)

        trained_models[name] = pipe
        leaderboard.append((name, acc))
        print(f"{name} Accuracy: {acc:.4f}")

    # Sort by accuracy descending
    leaderboard = sorted(leaderboard, key=lambda x: x[1], reverse=True)
    
    print("\nMODEL LEADERBOARD")
    for name, score in leaderboard:
        print(f"{name}: {score:.4f}")

    return trained_models, leaderboard

# --- 4. Visualization Functions ---
def plot_confusion_matrix(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



def plot_roc_curve(model, X_test, y_test):
    # predict_proba returns [prob_class_0, prob_class_1]
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label="ROC Curve (AUC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()



# --- 5. Main Pipeline Execution ---
def run_pipeline(dataset_path, target_column):
    print(f"Loading dataset: {dataset_path}")
    X, y = load_dataset(dataset_path, target_column)

    preprocessor = build_preprocessor(X)

    # Stratify ensures the train/test split has the same proportion of classes as the original data
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    print("\nComparing Models...\n")
    trained_models, leaderboard = compare_models(
        preprocessor, 
        X_train, 
        X_test, 
        y_train, 
        y_test
    )

    best_model_name = leaderboard[0][0]
    best_model = trained_models[best_model_name]

    print(f"\nBest Model: {best_model_name}")
    print("\nClassification Report:\n")
    preds = best_model.predict(X_test)
    print(classification_report(y_test, preds))

    print("\nGenerating Confusion Matrix...")
    plot_confusion_matrix(best_model, X_test, y_test)

    print("\nGenerating ROC Curve...")
    plot_roc_curve(best_model, X_test, y_test)

# To run the pipeline:
# run_pipeline("your_file.csv", "your_target_column")