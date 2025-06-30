import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from datetime import datetime

### ------------------- PARAMETERS -------------------
base_params = ["days", "days_m", "Bsr", "dB", "Kp", "Ap", "SN", "F10.7obs", "F10.7adj", "D"]
target = "storm_occurred"
lags = [1, 2, 3]

def add_lag_features(df, columns, lags):
    lagged = pd.concat(
        [df[columns].shift(lag).add_suffix(f"_lag{lag}") for lag in lags],
        axis=1
    )
    return pd.concat([df, lagged], axis=1)

### ------------------- LOAD DATA -------------------
train_df = pd.read_csv("training_data_edited.csv")
test_df = pd.read_csv("testing_data_edited.csv")

# Add lag features
train_df = add_lag_features(train_df, base_params, lags).dropna()
test_df = add_lag_features(test_df, base_params, lags).dropna()

# Ensure storm_occurred is treated as int
train_df[target] = train_df[target].astype(int)
test_df[target] = test_df[target].astype(int)

params = [f"{col}_lag{lag}" for col in base_params for lag in lags]

### ------------------- TRAIN MODEL -------------------
neg = sum(train_df[target] == 0)
pos = sum(train_df[target] == 1)
scale = neg / pos if pos > 0 else 1

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale,
    random_state=42
)
model.fit(train_df[params], train_df[target])

### ------------------- EVALUATE -------------------
predictions = model.predict(test_df[params])
probs = model.predict_proba(test_df[params])[:, 1]

print("Accuracy:", accuracy_score(test_df[target], predictions))
print("Classification Report:\n", classification_report(test_df[target], predictions))
print("ROC AUC:", roc_auc_score(test_df[target], probs))

### ------------------- PLOT FEATURE IMPORTANCE -------------------
importance_df = pd.DataFrame({
    "Feature": params,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance Score")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

### ------------------- USER PREDICTION -------------------
def predict_storm_probability(model, param_columns):
    print("Paste multiple rows of historical data (1 per line), in this format:")
    print("YYYY,MM,DD,days,days_m,Bsr,dB,Kp,ap,index,hour,Ap,SN,F10.7obs,F10.7adj,D,storm_occurred")
    print("Press Enter twice when you're done.\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line.strip())

    if len(lines) < 3:
        print("Error: You must enter at least 3 rows of historical data.")
        return

    try:
        all_values = [list(map(float, line.split(','))) for line in lines]
    except ValueError:
        print("Error: All values must be numeric.")
        return

    full_columns = [
        "YYYY", "MM", "DD", "days", "days_m", "Bsr", "dB",
        "Kp", "ap", "index", "hour", "Ap", "SN", "F10.7obs", "F10.7adj", "D", "storm_occurred"
    ]
    df = pd.DataFrame(all_values, columns=full_columns)

    if len(df) < 3:
        print("Need at least 3 days of history.")
        return

    future_date_str = input("Enter future prediction date (MM/DD/YYYY): ")
    try:
        future_date = datetime.strptime(future_date_str.strip(), "%m/%d/%Y")
    except ValueError:
        print("Invalid date format.")
        return

    # Build input features using last 3 rows of history
    history_df = df.iloc[-3:]

    future_row = pd.Series({col: 0.0 for col in full_columns})
    future_row["YYYY"] = future_date.year
    future_row["MM"] = future_date.month
    future_row["DD"] = future_date.day

    for lag in [1, 2, 3]:
        lag_row = history_df.iloc[-lag]
        for col in base_params:
            future_row[f"{col}_lag{lag}"] = lag_row[col]

    # Calculate time gap penalty
    last_date = datetime(
        int(history_df.iloc[-1]["YYYY"]),
        int(history_df.iloc[-1]["MM"]),
        int(history_df.iloc[-1]["DD"])
    )
    days_gap = (future_date - last_date).days

    # Get storm probability
    X = pd.DataFrame([future_row])[param_columns]
    prob = model.predict_proba(X)[0][1]

    # Degrade certainty based on how far the prediction is
    if days_gap > 0:
        decay_factor = 0.97 ** days_gap  # decay 3% per day ahead
        prob *= decay_factor

    percent = round(prob * 100, 2)

    # Certainty feedback
    if percent > 90:
        confidence = "Very certain"
    elif percent > 70:
        confidence = "Fairly confident"
    elif percent > 50:
        confidence = "Somewhat unsure"
    elif percent > 25:
        confidence = "Very unsure"
    elif percent > 10:
        confidence = "Very very unsure"
    else:
        confidence = "Basically impossible"

    print(f"\nPredicted geomagnetic storm probability on {future_date.strftime('%m/%d/%Y')}: {percent}% → {confidence}")

predict_storm_probability(model, params)
