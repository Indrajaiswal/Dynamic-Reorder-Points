import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# MAPE
# ============================================================

def calculate_mape(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    mask = y_true != 0

    if mask.sum() == 0:
        return 0.0

    return np.mean(
        np.abs(
            (
                y_true[mask]
                - y_pred[mask]
            )
            / y_true[mask]
        )
    ) * 100


# ============================================================
# FIND DEMAND COLUMN
# ============================================================

def find_demand_column(df):

    demand_aliases = [
        "units_sold",
        "quantity_sold",
        "qty_sold",
        "sales_quantity",
        "sales_qty",
        "quantity",
        "qty",
        "units",
        "demand",
        "sales"
    ]

    for column in demand_aliases:

        if column in df.columns:
            return column

    return None


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(df):

    data = df.copy()

    # ========================================================
    # TARGET
    # ========================================================

    demand_column = find_demand_column(
        data
    )

    if demand_column is None:

        raise ValueError(
            "Demand column not found."
        )

    # ========================================================
    # TARGET CLEANING
    # ========================================================

    y = pd.to_numeric(
        data[demand_column],
        errors="coerce"
    )

    y = y.fillna(0)

    # ========================================================
    # REMOVE TARGET
    # ========================================================

    X = data.drop(
        columns=[
            demand_column,
            "predicted_demand"
        ],
        errors="ignore"
    ).copy()

    # ========================================================
    # REMOVE RAW DATE
    # ========================================================

    datetime_columns = []

    for column in X.columns:

        if pd.api.types.is_datetime64_any_dtype(
            X[column]
        ):

            datetime_columns.append(
                column
            )

    X = X.drop(
        columns=datetime_columns,
        errors="ignore"
    )

    # ========================================================
    # REMOVE IDENTIFIER COLUMNS
    # ========================================================

    # SKU/product IDs are identifiers rather than
    # useful numerical relationships.

    identifier_columns = []

    for column in X.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
        )

        if normalized in [
            "product_id",
            "sku",
            "sku_id",
            "item_id"
        ]:

            identifier_columns.append(
                column
            )

    X = X.drop(
        columns=identifier_columns,
        errors="ignore"
    )

    # ========================================================
    # EXACT FEATURE LIST
    # ========================================================

    feature_columns = X.columns.tolist()

    if len(feature_columns) == 0:

        raise ValueError(
            "No usable features found for model training."
        )

    # ========================================================
    # NUMERIC FEATURES
    # ========================================================

    numeric_features = (
        X
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    categorical_features = (
        X
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    numeric_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "HistGradientBoosting":
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                random_state=42
            )
    }

    # ========================================================
    # STORAGE
    # ========================================================

    results = []

    trained_models = {}

    # ========================================================
    # TRAIN
    # ========================================================

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        # ----------------------------------------------------
        # HIST GRADIENT BOOSTING
        # ----------------------------------------------------

        if name == "HistGradientBoosting":

            model_features = numeric_features

            if len(model_features) == 0:

                print(
                    f"Skipping {name}: "
                    "no numeric features."
                )

                continue

            model_pipeline = Pipeline(
                steps=[

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "model",
                        model
                    )
                ]
            )

            X_train_model = X_train[
                model_features
            ]

            X_test_model = X_test[
                model_features
            ]

        # ----------------------------------------------------
        # LINEAR / RANDOM FOREST
        # ----------------------------------------------------

        else:

            model_features = feature_columns

            model_pipeline = Pipeline(
                steps=[

                    (
                        "preprocessor",
                        preprocessor
                    ),

                    (
                        "model",
                        model
                    )
                ]
            )

            X_train_model = X_train

            X_test_model = X_test

        # ----------------------------------------------------
        # FIT
        # ----------------------------------------------------

        model_pipeline.fit(
            X_train_model,
            y_train
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        predictions = model_pipeline.predict(
            X_test_model
        )

        predictions = np.maximum(
            predictions,
            0
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        mape = calculate_mape(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "Model": name,

            "MAE": round(
                mae,
                4
            ),

            "RMSE": round(
                rmse,
                4
            ),

            "MAPE (%)": round(
                mape,
                4
            )
        })

        trained_models[name] = {

            "model": model_pipeline,

            "features": model_features
        }

        print(
            f"{name} completed."
        )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise ValueError(
            "No models were successfully trained."
        )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    results_df = (
        results_df
        .sort_values(
            by="MAE"
        )
        .reset_index(
            drop=True
        )
    )

    best_model_name = (
        results_df
        .iloc[0]["Model"]
    )

    best_model = (
        trained_models[
            best_model_name
        ]["model"]
    )

    best_features = (
        trained_models[
            best_model_name
        ]["features"]
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 60
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nBest Model: "
        f"{best_model_name}"
    )

    print(
        f"\nNumber of features used: "
        f"{len(best_features)}"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "results":
            results_df,

        "best_model":
            best_model,

        "best_model_name":
            best_model_name,

        "feature_columns":
            best_features,

        "demand_column":
            demand_column,

        "trained_models":
            trained_models
    }