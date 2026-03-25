import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# LOAD DATA
df = pd.read_csv("winequality.csv.xls")

# CLEAN DATA
df = df.dropna()

# ENCODE TYPE COLUMN
df["type"] = df["type"].map({"red":0, "white":1})

# FEATURES / TARGET
X = df.drop("quality", axis=1)
y_raw = df["quality"]

# CONVERT QUALITY INTO CATEGORIES
y = pd.cut(
    y_raw,
    bins=[0,5,7,10],
    labels=["Low","Medium","Premium"]
)

# PIPELINE
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ))
])

# SPLIT
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

# TRAIN
pipe.fit(X_train,y_train)

# SAVE
joblib.dump(pipe,"model.pkl")
print("✅ model.pkl saved")