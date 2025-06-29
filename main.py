import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime



def add_lag_features(df, columns, lags):
    lagged = pd.concat(
        [df[columns].shift(lag).add_suffix(f"_lag{lag}") for lag in lags],
        axis=1
    )
    return pd.concat([df, lagged], axis=1)


# loading data
train_df = pd.read_csv("training_data.csv")             
test_df = pd.read_csv("testing_data.csv")

# define params and target
base_params = ["days", "days_m", "Bsr", "dB", "Kp", "Ap", "SN", "F10.7obs", "F10.7adj", "D"]
target = "storm_occurred"
lags = [1,2,3]

# add lag features
train_df = add_lag_features(train_df, base_params, lags).dropna()
test_df = add_lag_features(test_df, base_params, lags).dropna()

# generate list of lagged parameter names
params = [f"{col}_lag{lag}" for col in base_params for lag in lags]

# training the Random Forest AI model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_df[params], train_df[target])

# predict geomagnetic storms (yes or no)
predictions = model.predict(test_df[params])

# evaluate the model
print("Accuracy:", accuracy_score(test_df[target], predictions))
print("Classification Report:\n", classification_report(test_df[target], predictions))

# ranking parameter importance in predicting geomagnetic storms
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Parameter": params,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

# plot the importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Parameter"], importance_df["Importance"])
plt.xlabel("Importance Score")
plt.title("Parameter Importance in Predicting Geomagnetic Storms")
plt.tight_layout()
plt.show()


# USER INTERACTION ------------------- 
def predict_storm_probability(model, param_columns):
    print("Enter 3 consecutive days of data (1 row per day), in this format:")
    print("YYYY,MM,DD,days,days_m,Bsr,dB,Kp,Ap,SN,F10.7obs,F10.7adj,D:")
    print("-----------------------------------------------------------")

    rows = []
    for i in range(3):
        line = input(f"Day {i+1}: ")
        try:
            values = list(map(float, line.strip().split(',')))
            if len(values) != 13:
                print("Error: You must enter 13 values in this format:\nYYYY,MM,DD,days,days_m,Bsr,dB,Kp,Ap,SN,F10.7obs,F10.7adj,D")
                return
            rows.append(values)
        except ValueError:
            print("Error: All values must be numeric")
            return
        
    future_date_str = input("Enter future prediction date (MM/DD/YYYY): ")
    try:
        future_date = datetime.strptime(future_date_str.strip(), "%m/%d/%Y")
    except ValueError:
        print("Invalid date format.")
        return
    
    # create base dataframe from history (the lag days)
    full_columns = ["YYYY", "MM", "DD", "days", "days_m", "Bsr", "dB", "Kp", "Ap", 
                    "SN", "F10.7obs", "F10.7adj", "D"]
    history_df = pd.DataFrame(rows, columns=full_columns)

    # create a dummy row for the future date with only the date info
    future_row = pd.Series({col: 0.0 for col in full_columns})
    future_row["YYYY"] = future_date.year
    future_row["MM"] = future_date.month
    future_row["DD"] = future_date.day

    # add lag features to the future row using the history
    for lag in [1, 2, 3]:
        lag_row = history_df.iloc[-lag]
        for col in base_params:
            future_row[f"{col}_lag{lag}"] = lag_row[col]

    # format data for model
    X = pd.DataFrame([future_row])[param_columns]
    prob = model.predict_proba(X)[0][1]
    percent = round(prob * 100, 2)

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

    print(f"\nProbability of geomagnetic storm: {percent}% -> {confidence}")    

predict_storm_probability(model, params)