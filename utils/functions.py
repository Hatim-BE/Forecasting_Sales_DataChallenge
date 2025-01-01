import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from xgboost import XGBRegressor


def create_df(path):
    df = pd.read_csv(path)
    return df
### numerical / categorical columns
def get_num_cat_columns(df) -> dict:
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    return {"numerical" : numerical_columns, "categorical" : categorical_columns}
### nulls_summary_table
def nulls_summary_table(df):
    null_values = pd.DataFrame(df.isnull().sum())
    null_values[1] = null_values[0]/len(df)
    null_values.columns = ['null_count','null_percentage']
    return null_values
## B/ Transformations
### Drop columns
def drop_columns(df, columns):
    dropped = df.copy()
    dropped = dropped.drop(columns, axis=1)
    return dropped
### drop_null_values

def drop_null_values(df):
    dropped = df.copy()
    dropped = dropped.dropna()
    return dropped

def drop_null_values_columns(df, columns):
    dropped = df.copy()
    dropped = dropped.dropna(subset=columns)
    return dropped
### Mean/Median/Mode Imputation
def mean_median_mode_imputation(df, columns, option = "mean"):
  imputed_df = df.copy()
  for column in columns:
    if option == 'mean':
      imputed_df[column] = imputed_df[column].fillna(np.ceil(imputed_df[column].mean()))
    elif option == 'median':
      imputed_df[column] = imputed_df[column].fillna(imputed_df[column].median())
    elif option == 'mode':
      imputed_df[column] = imputed_df[column].fillna(imputed_df[column].mode()[0])

  return imputed_df
### Custom fill_na
def fill_weekend(df):
    df = df.copy()
    df['weekend'] = df['weekend'].fillna((df['date'].dt.weekday >= 5).astype(int))
    return df

def impute_jour_ferie_mode(df):
    jour_ferie_df = df.copy()
    # Fill NaN values in 'jour_ferie' by the mode (most frequent value) within each group of 'date'
    jour_ferie_df['jour_ferie'] = jour_ferie_df.groupby('date')['jour_ferie'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0))
    return jour_ferie_df

def impute_jour_ferie(df):
  jf = df.copy()
  jf['date'] = pd.to_datetime(jf['date'])  # Ensure 'date' is in datetime format
  # Step 1: Extract unique (day, month) -> jour_ferie mapping
  mapping_df = jf.dropna(subset=['jour_ferie'])[jf["jour_ferie"] == 1]  # Only use rows where 'jour_ferie' is not NaN
  jour_ferie_map = dict(zip(
      zip(mapping_df['date'].dt.day, mapping_df['date'].dt.month),
      mapping_df['jour_ferie']
  ))

  jf['jour_ferie'] = jf.apply(
      lambda row: jour_ferie_map.get((row['date'].day, row['date'].month), 0),
      axis=1
  )
  return jf


def impute_promotion_mode(df):
    promotion_df = df.copy()
    # Fill NaN values in 'promotion' by the mode (most frequent value) within each group of 'date'
    promotion_df['promotion'] = promotion_df.groupby('date')['promotion'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0))
    return promotion_df

def impute_promotion(df):
  promo = df.copy()
  # Ensure 'date' is in datetime format
  promo['date'] = pd.to_datetime(promo['date'], errors='coerce')

  # Step 1: Add 'day' and 'month' columns
  promo['day'] = promo['date'].dt.day
  promo['month'] = promo['date'].dt.month

  # Step 2: Extract the mapping for (day, month, jour_ferie) -> promotion
  promo_mapping = promo.dropna(subset=['promotion']).groupby(['day', 'month', 'jour_ferie'])['promotion'].max()
  promo_mapping = promo_mapping.to_dict()

  # Step 3: Impute missing 'promotion' using the mapping
  promo['promotion'] = promo.apply(
      lambda row: promo_mapping.get((row['day'], row['month'], row['jour_ferie']), 0)
      if pd.isna(row['promotion']) else row['promotion'],
      axis=1
  )
  return promo

def impute_meteo(df):
    meteo_df = df.copy()

    # Group by 'region' and 'moment_journee', and fill missing values using the mode
    meteo_df['condition_meteo'] = meteo_df.groupby(['date', 'region', 'moment_journee'])['condition_meteo'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

    return meteo_df


def impute_moment_journee(df):
    moment_journee_df = df.copy()
    # Fill NaN values in 'moment_journee' by the mode (most frequent value) within each group of 'date'
    moment_journee_df['moment_journee'] = moment_journee_df.groupby(['date', 'condition_meteo'])['moment_journee'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0))
    return moment_journee_df

def impute_column_randomly(df, column):
    # Get non-null values from the column
    non_null_values = df[column].dropna().values

    # Check if there are any non-null values to sample from
    if len(non_null_values) == 0:
        raise ValueError(f"No non-null values to sample from in column '{column}'.")

    # Randomly sample values for missing entries
    missing_indices = df[column].isna()
    df.loc[missing_indices, column] = np.random.choice(non_null_values, size=missing_indices.sum(), replace=True, random_state=42)

    return df

def replace_marque(row, marque_mapping):
    if pd.isna(row['marque']) and row['id_produit'] in marque_mapping:
        return marque_mapping[row['id_produit']][0]  # Replace with first marque if found
    return row['marque']  # Keep the original marque if not found


def ML_impute(df, option):
    df_imputed = df.copy()
    random_state = 42

    if option == "knn":
        # KNNImputer
        imputer = KNNImputer(n_neighbors=10)
        df_imputed[:] = imputer.fit_transform(df)

    elif option == "iterative":
        # IterativeImputer
        imputer = IterativeImputer(max_iter=10, random_state=random_state)
        df_imputed[:] = imputer.fit_transform(df)

    elif option == "random_forest":
        # Custom Imputer using Random Forest for numeric columns
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                # Separate known and unknown values
                train_data = df.loc[df[col].notnull()]
                test_data = df.loc[df[col].isnull()]

                X_train = train_data.drop(columns=[col])
                y_train = train_data[col]
                X_test = test_data.drop(columns=[col])

                if X_train.empty or X_test.empty: continue  # Skip if no data

                # Train RandomForestRegressor
                rf = RandomForestClassifier(random_state=random_state)
                rf.fit(X_train, y_train)
                preds = rf.predict(X_test)

                # Fill missing values
                df_imputed.loc[df[col].isnull(), col] = preds

    elif option == "mlp":
        # Custom Imputer using MLPRegressor (Neural Network) for numeric columns
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                train_data = df.loc[df[col].notnull()]
                test_data = df.loc[df[col].isnull()]

                X_train = train_data.drop(columns=[col])
                y_train = train_data[col]
                X_test = test_data.drop(columns=[col])

                if X_train.empty or X_test.empty: continue  # Skip if no data

                # Train MLPRegressor
                mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=random_state)
                mlp.fit(X_train, y_train)
                preds = mlp.predict(X_test)

                # Fill missing values
                df_imputed.loc[df[col].isnull(), col] = preds

    else:
        raise ValueError("Invalid option. Choose from 'simple', 'knn', 'iterative', 'random_forest', or 'mlp'.")

    return df_imputed


def impute_nums_ml(df, target_col, predictor_cols, model_type='linear'):
    """
    Impute missing values in a target column using a predictive model,
    ensuring predictors have no missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The column with missing values to be imputed.
        predictor_cols (list): List of columns to use as predictors.
        model_type (str): Type of model ('linear' or 'random_forest').

    Returns:
        pd.DataFrame: DataFrame with imputed values for the target column.
    """
    df = df.copy()

    # Separate rows where target_col is not null (training) and null (to impute)
    train_data = df[df[target_col].notna()]
    test_data = df[df[target_col].isna()]

    # Drop rows with missing predictors in training data
    train_data = train_data.dropna(subset=predictor_cols)

    # Features and target for training
    X_train = train_data[predictor_cols]
    y_train = train_data[target_col]

    # Drop rows with missing predictors in test data
    X_test = test_data[predictor_cols].dropna()

    # Check if X_train or X_test are empty
    if X_train.empty or X_test.empty:
        print(f"No sufficient data for training or predicting {target_col}")
        return df

    # Initialize the model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model_type. Use 'linear' or 'random_forest'.")

    # Train the model
    model.fit(X_train, y_train)

    # Predict missing values
    predicted_values = model.predict(X_test)
    print(f"number of predicted values {len(predicted_values)}")

    # Fill missing values in the target column
    df.loc[X_test.index, target_col] = predicted_values

    print(f"Missing values in '{target_col}' have been imputed using {model_type} model.")
    return df


def impute_catrgories_ml(df, target, predictors):
    """
    Impute missing values in a categorical column using ML classification (without encoding).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target (str): The name of the target column (categorical with missing values).
    predictors (list): List of predictor column names.

    Returns:
    pd.DataFrame: DataFrame with imputed target column values.
    """
    # Ensure that 'marque' is treated as a categorical column
    df[predictors] = df[predictors].astype(str)  # Convert to string type

    # Separate rows with and without the target value
    train_data = df[df[target].notna()].copy()
    test_data = df[df[target].isna()].copy()

    # Features and target
    X = train_data[predictors]  # Features (no encoding, raw data)
    y = train_data[target]  # Target column (no encoding, raw data)
    X_missing = test_data[predictors]  # Features for rows with missing target

    # Train-test split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train CatBoost Classifier (handles categorical features natively)
    model = CatBoostClassifier(iterations=50, depth=6, learning_rate=0.01, verbose=200)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_valid)
    # print("Model Evaluation:\n")
    # print(classification_report(y_valid, y_pred))

    # Predict missing values
    if not X_missing.empty:
        test_data[target] = model.predict(X_missing).ravel()
        # Merge back into the original DataFrame
        df.loc[df[target].isna(), target] = test_data[target]

    return df


### a/ Encoding
#### One Hot encoding
def one_hot_encode(df, columns, drop_f=False):
    encoded_df = df.copy()

    # Perform one-hot encoding
    encoded_df = pd.get_dummies(encoded_df, columns=columns, drop_first=drop_f, dtype=int)

    # Identify new columns (those that were added by the one-hot encoding process)
    new_columns = [col for col in encoded_df.columns if col not in df.columns]
    print(new_columns)

    return encoded_df, new_columns

#### Label Encoding

def label_encode(df, columns):
    encoded_df = df.copy()
    label_encoder = LabelEncoder()
    for col in columns:
        encoded_df[col] = label_encoder.fit_transform(encoded_df[col])
    return encoded_df
#### Ordinal Encoding
#### Target encoding
def target_encode(df, columns, target):
    """
    Encode a categorical column using target encoding (mean of the target variable).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the categorical column to encode.
    target (str): The name of the target column to use for encoding.

    Returns:
    pd.DataFrame: DataFrame with the target-encoded column.
    """
    # Calculate the mean of the target variable for each category
    encoding = df.groupby(columns)[target].mean()

    # Map the encoding back to the original DataFrame
    df[columns] = df[columns].map(encoding)

    return df

#### Custom encoding
'''
  encoding 'id_produit'
'''
def encode_id_produit(df):
  encoded_df = df.copy()
  tag1 = pd.DataFrame(encoded_df['id_produit'].str.split('-').str[0])
  tag2 = pd.DataFrame(encoded_df['id_produit'].str.split('-').str[1])
  tag3 = pd.DataFrame(encoded_df['id_produit'].str.split('-').str[2])

  tag1 = label_encode(tag1, ["id_produit"])
  encoded_df.loc[:, 'id_produit'] = (tag1['id_produit'].astype(str) +
                                     tag2['id_produit'].astype(str) +
                                     tag3['id_produit'].astype(str))
  encoded_df['id_produit'] = encoded_df['id_produit'].astype(int)
  return encoded_df

def frequency_encoding(df, column_name):
    """
    Perform frequency encoding on a categorical column.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: The name of the column to encode.

    Returns:
    - The DataFrame with the frequency-encoded column.
    """
    # Calculate the frequency of each category in the column
    freq_encoding = df[column_name].value_counts() / len(df) * 100

    # Map the frequency encoding to the original column
    df[column_name] = df[column_name].map(freq_encoding)

    return df

def ordinal_encode(df, column):
    encoded_df = df.copy()
    ordinal_encoder = OrdinalEncoder()
    encoded_df[column] = ordinal_encoder.fit_transform(encoded_df[[column]])
    return encoded_df


def feature_scaling(df, columns, option):
    scaled_df = df.copy()
    if option == "maxabs":
        scaler = MaxAbsScaler()
    elif option == "minmax":
        scaler = MinMaxScaler()
    elif option == "standard":
        scaler = StandardScaler()
    elif option == "robust":
        scaler = RobustScaler()

    scaled_df[columns] = scaler.fit_transform(scaled_df[columns])
    return scaled_df
## C/ EDA
### Box plots
def show_box_plots(df, columns):
    n_cols = 3
    n_rows = (len(columns) // n_cols) + (1 if len(columns) % n_cols != 0 else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        df[col].plot(kind='box', ax=ax)
        ax.set_title(f'Box Plot of {col}')
        ax.set_ylabel('Value')

    # Hide any unused subplots (if there are fewer columns than subplots)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
### extract_outliers
def extract_outliers(df):
    ## Get IQR
    iqr_q1 = np.quantile(df, 0.25)
    iqr_q3 = np.quantile(df, 0.75)
    med = np.median(df)

    # finding the iqr region
    iqr = iqr_q3-iqr_q1

    # finding upper and lower whiskers
    upper_bound = iqr_q3+(1.5*iqr)
    lower_bound = iqr_q1-(1.5*iqr)

    outliers = df[(df <= lower_bound) | (df >= upper_bound)]
    print('Outliers within the box plot are :{}'.format(outliers))
    return outliers
### Historgram plots
def show_hist_plots(df, columns):
    n_cols = 3
    n_rows = (len(columns) // n_cols) + (1 if len(columns) % n_cols != 0 else 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(df[col], ax=ax)
        ax.set_title(f'Histogram of {col}')
        ax.set_ylabel('Value')

    # Hide any unused subplots (if there are fewer columns than subplots)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def plot_histograms(df_train, df_test, target_col, n_cols=3):
    n_rows = (len(df_train.columns) - 1) // n_cols + 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()

    for i, var_name in enumerate(df_train.columns.tolist()):
        #print(var_name)
        ax = axes[i]
        sns.distplot(df_train[var_name], kde=True, ax=ax, label='Train')      # plot train data

        if var_name != target_col:
            sns.distplot(df_test[var_name], kde=True, ax=ax, label='Submission')    # plot test data
        ax.set_title(f'{var_name} distribution')
        ax.legend()

    plt.tight_layout()
    plt.show()
### Outliers capping
def winsorize(df, column, upper = 75, lower = 25):
    capped_df = df.copy()
    col_df = capped_df[column]

    perc_upper = np.percentile(capped_df[column],upper)
    perc_lower = np.percentile(capped_df[column],lower)

    capped_df[column] = np.where(capped_df[column] >= perc_upper,
                          perc_upper,
                          capped_df[column])

    capped_df[column] = np.where(capped_df[column] <= perc_lower,
                          perc_lower,
                          capped_df[column])

    return capped_df
## D/ Cross validation


def tr_te_split(df, test_size = 0.2):
  train, test = train_test_split(df, test_size=test_size, random_state=42)
  return train, test

def KFold_split(df, n_splits = 5):
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

  for train_index, test_index in kf.split(df):
    train = df.iloc[train_index]
    test = df.iloc[test_index]
    yield train, test

def leave_one_out_split(df):
  loo = LeaveOneOut()
  # Iterate through each split
  for train_index, test_index in loo.split(df):
      train = df.iloc[train_index]
      test = df.iloc[test_index]
      yield train, test


def time_series_kfold_split(df, n_splits=5):
  tscv = TimeSeriesSplit(n_splits=n_splits)

  # Iterate through each split
  for train_index, test_index in tscv.split(df):
      train = df.iloc[train_index]
      test = df.iloc[test_index]
      yield train, test

## E/ Metrics
# Define MAPE function
def mape_function(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))