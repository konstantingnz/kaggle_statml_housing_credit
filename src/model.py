import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_logistic_regression():
    """
    Train logistic regression models using baseline features, PCA, and SVD, 
    then generate predictions and save results.
    """
    # 1️⃣ Load datasets
    train_df = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_final.csv")
    test_df = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_final.csv")

    # 2️⃣ Separate features and target variable
    X_train = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    y_train = train_df["TARGET"]
    X_test = test_df.drop(columns=["SK_ID_CURR"])

    # 3️⃣ Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4️⃣ PCA (dimensionality reduction)
    pca = PCA(n_components=0.95)  # Keep 95% of the variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 5️⃣ SVD (alternative for sparse data)
    svd = TruncatedSVD(n_components=50)  # Reduce to 50 dimensions
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)

    # 6️⃣ Train Logistic Regression (Baseline, PCA, SVD)
    models = {
        "Baseline": (X_train_scaled, X_test_scaled),
        "PCA": (X_train_pca, X_test_pca),
        "SVD": (X_train_svd, X_test_svd)
    }

    submissions = {}

    for name, (X_train_model, X_test_model) in models.items():
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_model, y_train)
        
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]
        
        # Store predictions
        submissions[name] = pd.DataFrame({"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": y_pred_proba})
        submissions[name].to_csv(f"/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/results/submission_logistic_regression_{name}.csv", index=False)
        
        print(f"{name} - Predictions saved.")

def train_random_forest():
    """
    Train a Random Forest model on the dataset and generate predictions for submission.
    """
    # 1️⃣ Load datasets
    train_df = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_final.csv")
    test_df = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_final.csv")

    # 2️⃣ Separate features and target variable
    X_train = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    y_train = train_df["TARGET"]
    X_test = test_df.drop(columns=["SK_ID_CURR"])

    # 3️⃣ Train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,  # Number of trees (adjustable)
        max_depth=10,  # Maximum depth (prevents overfitting)
        random_state=42,
        n_jobs=-1,  # Use all CPU cores for faster training
        class_weight="balanced"  # Handle class imbalance
    )

    rf_model.fit(X_train, y_train)

    # 4️⃣ Predictions on test dataset
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    # 5️⃣ Generate submission file
    submission_rf = pd.DataFrame({"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": y_pred_proba})
    submission_rf.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/results/submission_RandomForest.csv", index=False)

    print("✅ Random Forest - Predictions saved.")

def train_xgboost():
    """
    Train an XGBoost model on the dataset and generate predictions for submission.
    """
    # 1️⃣ Load datasets
    train_df = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_final.csv")
    test_df = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_final.csv")

    # 2️⃣ Separate features and target variable
    X_train = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    y_train = train_df["TARGET"]
    X_test = test_df.drop(columns=["SK_ID_CURR"])

    # 3️⃣ Configure XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,  # Number of trees
        learning_rate=0.05,  # Learning rate
        max_depth=6,  # Tree depth
        subsample=0.8,  # Data sampling to prevent overfitting
        colsample_bytree=0.8,  # Feature sampling for each tree
        objective="binary:logistic",  # Binary classification problem
        eval_metric="auc",  # Optimize ROC-AUC
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )

    # 4️⃣ Train the model
    xgb_model.fit(X_train, y_train)

    # 5️⃣ Predictions on test dataset
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    # 6️⃣ Generate submission file
    submission_xgb = pd.DataFrame({"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": y_pred_proba})
    submission_xgb.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/results/submission_XGBoost.csv", index=False)

    print("✅ XGBoost - Predictions saved.")