import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def process_application_train():
    # Load data
    df_train = pd.read_csv('/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train.csv')
    df_test = pd.read_csv('/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test.csv')
    
    # Remove columns with more than 30% missing values
    missing_threshold = 0.3
    missing_percentage = df_train.isnull().sum() / len(df_train)
    columns_to_drop = missing_percentage[missing_percentage > missing_threshold].index
    df_train = df_train.drop(columns=columns_to_drop)

    # Fill missing values (train only)
    numerical_columns = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df_train[numerical_columns] = df_train[numerical_columns].fillna(df_train[numerical_columns].mean())

    categorical_columns = df_train.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])

    # One-Hot Encoding (train only)
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_columns, drop_first=True)

    # Separate features (X) and target (y)
    X = df_train_encoded.drop(columns=["TARGET"])
    y = df_train_encoded["TARGET"]

    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Feature importance analysis
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()

    # Select the most important features (80% threshold)
    selected_features = importance_df[importance_df['Cumulative_Importance'] <= 0.80]['Feature'].tolist()

    # Filter the training dataset with selected features
    df_train_encoded = df_train_encoded[['SK_ID_CURR', 'TARGET'] + selected_features]

    # ===============================
    # PROCESSING df_test AFTER FEATURE SELECTION
    # ===============================

    # Fill missing values in df_test
    numerical_columns = df_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df_test.select_dtypes(include=['object']).columns
    df_test[numerical_columns] = df_test[numerical_columns].fillna(df_test[numerical_columns].mean())
    
    for col in categorical_columns:
        df_test[col] = df_test[col].fillna(df_test[col].mode()[0])

    # One-Hot Encoding for df_test (after feature selection)
    df_test_encoded = pd.get_dummies(df_test, columns=categorical_columns, drop_first=True)

    df_test_encoded = df_test_encoded[['SK_ID_CURR'] + selected_features]  # Keep only selected features

    # Save the processed files
    df_train_encoded.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_selected.csv", index=False)
    df_test_encoded.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_selected.csv", index=False)

    print("Files have been saved:")
    print("- 'application_train_selected.csv'")
    print("- 'application_test_selected.csv'")
def process_bureau():
    # Load data
    bureau_balance = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/bureau_balance.csv")
    bureau = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/bureau.csv")
    application_train = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train.csv")

    # Clean and process bureau
    bureau = bureau.drop(columns=bureau.columns[bureau.isna().mean() > 0.3])
    num_cols = bureau.select_dtypes(include=["number"]).columns
    cat_cols = bureau.select_dtypes(include=["object"]).columns

    bureau[num_cols] = bureau[num_cols].fillna(bureau[num_cols].mean())
    for col in cat_cols:
        bureau[col] = bureau[col].fillna(bureau[col].mode()[0])

    # Encode categorical variables
    for col in cat_cols:
        if bureau[col].nunique() <= 15:
            bureau = pd.get_dummies(bureau, columns=[col], prefix=col)
        else:
            bureau[col] = LabelEncoder().fit_transform(bureau[col])

    # Merge with application_train
    df_rf = bureau.merge(application_train[["SK_ID_CURR", "TARGET"]], on="SK_ID_CURR", how="left").dropna(subset=["TARGET"])

    # Prepare data
    X = df_rf.drop(columns=["SK_ID_CURR", "TARGET"])
    y = df_rf["TARGET"]

    # Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # Select the 6 most important features
    important_features = feature_importance["feature"].tolist()[:6]
    bureau = bureau[["SK_ID_CURR"] + important_features]

    # Add STATUS_SCORE variable in bureau_balance
    status_mapping = {'0': 0, 'C': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    bureau_balance["STATUS_SCORE"] = bureau_balance["STATUS"].map(status_mapping).fillna(0)

    # Aggregate by SK_ID_BUREAU
    bureau_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
        max_delay=("STATUS_SCORE", "max"),
        avg_delay=("STATUS_SCORE", "mean"),
        unpaid_ratio=("STATUS_SCORE", lambda x: (x > 0).sum() / len(x))
    ).reset_index()

    # Merge with bureau
    bureau = bureau.merge(bureau_agg, on="SK_ID_BUREAU", how="left").fillna(0)

    # Aggregate by SK_ID_CURR
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

    # Save processed data
    bureau.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/bureau_bureau_balance-selected.csv", index=False)

    print("Files have been saved:")
    print("- 'bureau_bureau_balance-selected.csv'")


def process_right_data():
    # Load the data
    POS_CASH_balance = pd.read_csv('/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/POS_CASH_balance.csv')
    credit_card_balance = pd.read_csv('/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/credit_card_balance.csv')
    installments_payments = pd.read_csv('/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/installments_payments.csv')
    files = [POS_CASH_balance, credit_card_balance, installments_payments]
    files_cleaned = []

    for file in files:
        # 1️⃣ Remove columns with more than 30% missing values
        threshold = 0.30  # 30% threshold
        nan_percent = file.isnull().mean()
        columns_to_drop = nan_percent[nan_percent > threshold].index
        file_cleaned = file.drop(columns=columns_to_drop)

        # 2️⃣ Handle remaining NaNs
        for col in file_cleaned.columns:
            if file_cleaned[col].dtype == "object":  # Categorical
                file_cleaned[col].fillna(file_cleaned[col].mode()[0], inplace=True)
            else:  # Numerical
                file_cleaned[col].fillna(file_cleaned[col].median(), inplace=True)

    POS_CASH_balance_cleaned, credit_card_balance_cleaned, installments_payments_cleaned = files_cleaned
    POS_CASH_balance_cleaned.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/POS_CASH_balance_cleaned.csv", index=False)
    credit_card_balance_cleaned.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/credit_card_balance_cleaned.csv", index=False)
    installments_payments_cleaned.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/installments_payments_cleaned.csv", index=False)

def process_right_data2():
    # Load the data
    previous_application = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/previous_application.csv")
    application_train = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train.csv", usecols=["SK_ID_CURR", "TARGET"])

    # Remove columns with more than 30% missing values
    threshold = 0.30  # 30% threshold
    nan_percent = previous_application.isnull().mean()
    columns_to_drop = nan_percent[nan_percent > threshold].index
    previous_application_cleaned = previous_application.drop(columns=columns_to_drop)

    # Handle remaining NaN values
    for col in previous_application_cleaned.columns:
        if previous_application_cleaned[col].dtype == "object":  # Categorical columns
            previous_application_cleaned[col].fillna(previous_application_cleaned[col].mode()[0], inplace=True)
    else:  # Numerical columns
        previous_application_cleaned[col].fillna(previous_application_cleaned[col].median(), inplace=True)

    # Merge to retrieve the target variable
    data = previous_application_cleaned.merge(application_train, on="SK_ID_CURR", how="left").dropna(subset=["TARGET"])

    # Separate features and target variable
    X = data.drop(columns=["SK_ID_CURR", "SK_ID_PREV", "TARGET"])
    y = data["TARGET"]

    # Encode categorical variables
    X = pd.get_dummies(X)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Take a 20% sample of the dataset for training
    X_train_sample, _, y_train_sample, _ = train_test_split(X, y, train_size=0.2, random_state=42, stratify=y)

    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_sample, y_train_sample)

    # Compute feature importance
    feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Select only the most relevant features
    id_columns = ["SK_ID_PREV", "SK_ID_CURR"]
    selected_features = feature_importances[feature_importances["Importance"] > 0.005]["Feature"].values

    # One-hot encode categorical variables
    previous_application_cleaned = pd.get_dummies(previous_application_cleaned)

    # Keep only the ID columns and selected features
    previous_application_filtered = previous_application_cleaned[id_columns + list(selected_features)]

    # Save the filtered dataset
    previous_application_filtered.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/previous_application_filtered.csv", index=False)

def merge_right_data():
    """Merges and aggregates data from various files to prepare the final dataset."""

    def aggregate_monthly_data(df, group_col, dataset_name):
        """Aggregates monthly data by computing useful statistics based on the dataset type."""
        
        agg_funcs_dict = {
            'pos_cash': {
                'MONTHS_BALANCE': ['min', 'max', 'mean'],
                'CNT_INSTALMENT': ['mean', 'median', 'max'],
                'CNT_INSTALMENT_FUTURE': ['mean', 'median', 'min'],
                'SK_DPD': ['mean', 'max'],
                'SK_DPD_DEF': ['mean', 'max']
            },
            'credit_card': {
                'MONTHS_BALANCE': ['min', 'max', 'mean'],
                'AMT_BALANCE': ['mean', 'max'],
                'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],
                'SK_DPD': ['mean', 'max'],
                'SK_DPD_DEF': ['mean', 'max']
            },
            'installments': {
                'NUM_INSTALMENT_VERSION': ['nunique'],
                'NUM_INSTALMENT_NUMBER': ['max'],
                'DAYS_INSTALMENT': ['mean', 'min', 'max'],
                'DAYS_ENTRY_PAYMENT': ['mean', 'min', 'max'],
                'AMT_INSTALMENT': ['sum', 'mean'],
                'AMT_PAYMENT': ['sum', 'mean']
            }
        }

        agg_funcs = agg_funcs_dict.get(dataset_name, {})
        available_agg_funcs = {col: funcs for col, funcs in agg_funcs.items() if col in df.columns}

        if available_agg_funcs:
            aggregated_df = df.groupby(group_col).agg(available_agg_funcs)
            aggregated_df.columns = ["_".join(col) for col in aggregated_df.columns]
            aggregated_df.reset_index(inplace=True)
            return aggregated_df
        else:
            return pd.DataFrame(columns=[group_col])  

    # Load datasets
    previous_application = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/previous_application.csv")
    pos_cash = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/POS_CASH_balance_cleaned.csv")
    credit_card = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/credit_card_balance_cleaned.csv")
    installments = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/installments_payments_cleaned.csv")

    # Aggregate data
    pos_cash_agg = aggregate_monthly_data(pos_cash, 'SK_ID_PREV', 'pos_cash')
    credit_card_agg = aggregate_monthly_data(credit_card, 'SK_ID_PREV', 'credit_card')
    installments_agg = aggregate_monthly_data(installments, 'SK_ID_PREV', 'installments')

    # Merge with previous_application
    final_df = previous_application.merge(pos_cash_agg, on='SK_ID_PREV', how='left')
    final_df = final_df.merge(credit_card_agg, on='SK_ID_PREV', how='left')
    final_df = final_df.merge(installments_agg, on='SK_ID_PREV', how='left')

    # Handle NaN values
    amount_cols = [col for col in final_df.columns if 'AMT_' in col]
    final_df[amount_cols] = final_df[amount_cols].fillna(0)

    mean_cols = [col for col in final_df.columns if '_mean' in col or '_median' in col]
    final_df[mean_cols] = final_df[mean_cols].apply(lambda x: x.fillna(x.median()))

    days_cols = [col for col in final_df.columns if 'DAYS_' in col]
    final_df[days_cols] = final_df[days_cols].apply(lambda x: x.fillna(x.median()))

    max_min_cols = [col for col in final_df.columns if '_max' in col or '_min' in col]
    for col in max_min_cols:
        if final_df[col].isnull().sum() > 0:  
            if '_max' in col:
                final_df[col].fillna(final_df[col].dropna().max(), inplace=True)
            elif '_min' in col:
                final_df[col].fillna(final_df[col].dropna().min(), inplace=True)

    final_df['NUM_INSTALMENT_VERSION_nunique'].fillna(0, inplace=True)

    # Drop unnecessary columns
    credit_card_cols = [
        'MONTHS_BALANCE_min', 'MONTHS_BALANCE_max', 'MONTHS_BALANCE_mean',
        'AMT_BALANCE_mean', 'AMT_BALANCE_max',
        'AMT_CREDIT_LIMIT_ACTUAL_mean', 'AMT_CREDIT_LIMIT_ACTUAL_max',
        'SK_DPD_mean', 'SK_DPD_max',
        'SK_DPD_DEF_mean', 'SK_DPD_DEF_max'
    ]
    cols_to_drop = [col for col in credit_card_cols if col in final_df.columns]
    final_df.drop(columns=cols_to_drop, inplace=True)

    # Aggregate at the individual level
    agg_dict = {
        'DAYS_DECISION': 'min',
        'HOUR_APPR_PROCESS_START': 'min',
        'AMT_ANNUITY': 'mean',
        'AMT_CREDIT': 'mean',
        'AMT_APPLICATION': 'mean',
        'AMT_GOODS_PRICE': 'mean',
        'SELLERPLACE_AREA': 'median',
        'CNT_PAYMENT': 'mean',
        'WEEKDAY_APPR_PROCESS_START_THURSDAY': 'sum',
        'WEEKDAY_APPR_PROCESS_START_MONDAY': 'sum',
        'WEEKDAY_APPR_PROCESS_START_FRIDAY': 'sum',
        'WEEKDAY_APPR_PROCESS_START_TUESDAY': 'sum',
        'WEEKDAY_APPR_PROCESS_START_WEDNESDAY': 'sum',
        'WEEKDAY_APPR_PROCESS_START_SATURDAY': 'sum',
        'WEEKDAY_APPR_PROCESS_START_SUNDAY': 'sum',
        'NAME_CLIENT_TYPE_Repeater': 'sum',
        'NAME_CLIENT_TYPE_Refreshed': 'sum',
        'MONTHS_BALANCE_min_x': 'min',
        'MONTHS_BALANCE_max_x': 'max',
        'MONTHS_BALANCE_mean_x': 'mean',
        'CNT_INSTALMENT_mean': 'mean',
        'CNT_INSTALMENT_median': 'median',
        'CNT_INSTALMENT_max': 'max',
        'CNT_INSTALMENT_FUTURE_mean': 'mean',
        'CNT_INSTALMENT_FUTURE_median': 'median',
        'CNT_INSTALMENT_FUTURE_min': 'min',
        'SK_DPD_mean_x': 'mean',
        'SK_DPD_max_x': 'max',
        'SK_DPD_DEF_mean_x': 'mean',
        'SK_DPD_DEF_max_x': 'max',
        'MONTHS_BALANCE_min_y': 'min',
        'MONTHS_BALANCE_max_y': 'max',
        'MONTHS_BALANCE_mean_y': 'mean',
        'SK_DPD_mean_y': 'mean',
        'SK_DPD_max_y': 'max',
        'SK_DPD_DEF_mean_y': 'mean',
        'SK_DPD_DEF_max_y': 'max',
        'NUM_INSTALMENT_VERSION_nunique': 'sum',
        'NUM_INSTALMENT_NUMBER_max': 'max',
        'DAYS_INSTALMENT_mean': 'mean',
        'DAYS_INSTALMENT_min': 'min',
        'DAYS_INSTALMENT_max': 'max',
        'DAYS_ENTRY_PAYMENT_mean': 'mean',
        'DAYS_ENTRY_PAYMENT_min': 'min',
        'DAYS_ENTRY_PAYMENT_max': 'max',
        'AMT_INSTALMENT_sum': 'sum',
        'AMT_INSTALMENT_mean': 'mean',
        'AMT_PAYMENT_sum': 'sum',
        'AMT_PAYMENT_mean': 'mean'
    }

    final_df_grouped = final_df.drop(columns=['SK_ID_PREV']).groupby('SK_ID_CURR').agg(agg_dict).reset_index()

    # Save the final dataset
    final_df_grouped.to_csv("final_application_data2.csv", index=False)


def merge():
    # Load the data
    application_train_selected = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_selected.csv")
    application_test_selected = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_selected.csv")
    bureau_selected = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/bureau_bureau_balance-selected.csv")
    final_application = pd.read_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/final_application_data2.csv")

    # Merge while keeping all rows from application_train_selected and application_test_selected
    df = application_train_selected.merge(bureau_selected, on="SK_ID_CURR", how="left")
    df = df.merge(final_application, on="SK_ID_CURR", how="left")

    df_test = application_test_selected.merge(bureau_selected, on="SK_ID_CURR", how="left")
    df_test = df_test.merge(final_application, on="SK_ID_CURR", how="left")

    # Replace missing values with 0
    df = df.fillna(0)
    df_test = df_test.fillna(0)

    # Save the merged files
    df.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_train_final.csv", index=False)
    df_test.to_csv("/Users/konstantinganz/Library/CloudStorage/OneDrive-Personnel/Documents/Konstantin/CS/S8 Hong Kong/Cours/Stat ML/Projet_1/kaggle_statml_housing_credit/data/home-credit-default-risk/application_test_final.csv", index=False)

    print("Files have been saved:")
    print("- 'application_train_final.csv'")
    print("- 'application_test_final.csv'")