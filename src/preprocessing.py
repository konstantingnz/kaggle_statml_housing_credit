import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def process_application_train():
    # Charger les données
    df_train = pd.read_csv('/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train.csv')
    df_test = pd.read_csv('/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test.csv')
    
    # Supprimer les colonnes avec plus de 30% de valeurs manquantes
    seuil_na = 0.3
    missing_percentage = df_train.isnull().sum() / len(df_train)
    columns_to_drop = missing_percentage[missing_percentage > seuil_na].index
    df_train = df_train.drop(columns=columns_to_drop)

    # Remplir les valeurs manquantes (train uniquement)
    numerical_columns = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_train[numerical_columns] = df_train[numerical_columns].fillna(df_train[numerical_columns].mean())

    categorical_columns = df_train.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

    # Encodage One-Hot (train uniquement)
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_columns, drop_first=True)

    # Séparer les caractéristiques (X) et la cible (y)
    X = df_train_encoded.drop(columns=["TARGET"])
    y = df_train_encoded["TARGET"]

    # Entraîner un modèle Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Importance des features
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()

    # Sélectionner les features les plus importantes (seuil de 80%)
    selected_features = importance_df[importance_df['Cumulative_Importance'] <= 0.80]['Feature'].tolist()

    # Filtrer le dataset train avec les features sélectionnées
    df_train_encoded = df_train_encoded[['SK_ID_CURR', 'TARGET'] + selected_features]

    # ===============================
    # TRAITEMENT DE df_test APRES SELECTION DES FEATURES
    # ===============================

    # Remplir les valeurs manquantes dans df_test
    numerical_columns = df_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df_test.select_dtypes(include=['object']).columns
    df_test[numerical_columns] = df_test[numerical_columns].fillna(df_test[numerical_columns].mean())
    
    for col in categorical_columns:
        df_test[col] = df_test[col].fillna(df_test[col].mode()[0])

    # Encodage One-Hot pour df_test (après sélection des features)
    df_test_encoded = pd.get_dummies(df_test, columns=categorical_columns, drop_first=True)

    df_test_encoded = df_test_encoded[['SK_ID_CURR'] + selected_features]  # Garder uniquement les bonnes features

    # Sauvegarder les fichiers
    df_train_encoded.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_selected.csv", index=False)
    df_test_encoded.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_selected.csv", index=False)

    print("Les fichiers ont été sauvegardés :")
    print("- 'application_train_selected.csv'")
    print("- 'application_test_selected.csv'")

def process_bureau():
    # Charger les données
    bureau_balance = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/bureau_balance.csv")
    bureau = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/bureau.csv")
    application_train = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train.csv")

    # Nettoyage et traitement de bureau
    bureau = bureau.drop(columns=bureau.columns[bureau.isna().mean() > 0.3])
    num_cols = bureau.select_dtypes(include=["number"]).columns
    cat_cols = bureau.select_dtypes(include=["object"]).columns

    bureau[num_cols] = bureau[num_cols].fillna(bureau[num_cols].mean())
    for col in cat_cols:
        bureau[col] = bureau[col].fillna(bureau[col].mode()[0])

    # Encodage des variables catégoriques
    for col in cat_cols:
        if bureau[col].nunique() <= 15:
            bureau = pd.get_dummies(bureau, columns=[col], prefix=col)
        else:
            bureau[col] = LabelEncoder().fit_transform(bureau[col])

    # Merge avec application_train
    df_rf = bureau.merge(application_train[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR", how="left").dropna(subset=["TARGET"])

    # Préparation des données
    X = df_rf.drop(columns=["SK_ID_CURR", "TARGET"])
    y = df_rf["TARGET"]
    

    # Modèle Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Importance des features
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # Sélection des 6 features les plus importantes
    important_features = feature_importance["feature"].tolist()[:6]
    bureau = bureau[["SK_ID_CURR"] + important_features]

    # Ajout de la variable STATUS_SCORE dans bureau_balance
    status_mapping = {'0': 0, 'C': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    bureau_balance["STATUS_SCORE"] = bureau_balance["STATUS"].map(status_mapping).fillna(0)

    # Agrégation par SK_ID_BUREAU
    bureau_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
        max_delay=("STATUS_SCORE", "max"),
        avg_delay=("STATUS_SCORE", "mean"),
        unpaid_ratio=("STATUS_SCORE", lambda x: (x > 0).sum() / len(x))
    ).reset_index()

    # Fusion avec bureau
    bureau = bureau.merge(bureau_agg, on="SK_ID_BUREAU", how="left").fillna(0)

    # Agrégation par SK_ID_CURR
    bureau = bureau.groupby("SK_ID_CURR").agg(
        max_delay=("max_delay", "max"),
        avg_delay=("avg_delay", "mean"),
        unpaid_ratio=("unpaid_ratio", "mean"),
        DAYS_CREDIT=("DAYS_CREDIT", "min"),
        DAYS_CREDIT_ENDDATE=("DAYS_CREDIT_ENDDATE", "max"),
        DAYS_CREDIT_UPDATE=("DAYS_CREDIT_UPDATE", "max"),
        AMT_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        AMT_CREDIT_SUM_DEBT=("AMT_CREDIT_SUM_DEBT", "sum")
    ).reset_index()

    # Sauvegarde des données traitées
    bureau.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/bureau_bureau_balance-selected.csv", index=False)

    print("Les fichiers ont été sauvegardés :")
    print("- 'bureau_bureau_balance.csv'")

def merge():
    # Charger les données
    application_train_selected = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_selected.csv")
    application_test_selected = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_selected.csv")
    bureau_selected = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/bureau_bureau_balance-selected.csv")
    final_application = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/final_application_data2.csv")

    # Fusionner en conservant toutes les lignes de application_train_selected et application_test_selected
    df = application_train_selected.merge(bureau_selected, on="SK_ID_CURR", how="left")
    df = df.merge(final_application, on="SK_ID_CURR", how="left")

    df_test = application_test_selected.merge(bureau_selected, on="SK_ID_CURR", how="left")
    df_test = df_test.merge(final_application, on="SK_ID_CURR", how="left")

    # Remplacer les valeurs manquantes par 0
    df = df.fillna(0)
    df_test = df_test.fillna(0)


    # Sauvegarder le fichier fusionné
    df.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_final.csv", index=False)
    df_test.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_final.csv", index=False)

    print("Les fichiers ont été sauvegardés:")
    print("- 'application_train_final.csv'")
    print("- 'application_test_final.csv'")


merge()
