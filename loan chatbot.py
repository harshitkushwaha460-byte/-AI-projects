import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

df = pd.read_csv(
    "loan_dataset.csv"
)

X = df.drop(
    ["Loan_Status", "Age"],
    axis=1
)

y = df["Loan_Status"]

categorical_features = [
    "Gender",
    "Marital_Status",
    "Education",
    "Employment_Type"
]

numeric_features = [
    "Years_Employed",
    "Applicant_Income",
    "CoApplicant_Income",
    "CIBIL_Score",
    "Existing_Debt",
    "Property_Value",
    "Loan_Amount",
    "Loan_Term"
]

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore"
        ),
        categorical_features
    ),
    (
        "num",
        "passthrough",
        numeric_features
    )
])

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline.fit(
    X_train,
    y_train
)

pred = pipeline.predict(
    X_test
)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        pred
    )
)

with open(
    "loan_pipeline.pkl",
    "wb"
) as f:

    pickle.dump(
        pipeline,
        f
    )

print("Pipeline Saved")