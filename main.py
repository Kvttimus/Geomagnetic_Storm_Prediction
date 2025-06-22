import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# loading data
train_df = pd.read_csv("training_data.csv")
test_df = pd.read_csv("testing_data.csv")

# define params and target
params = ["days", "days_m", "Bsr", "dB", "Kp", "Ap", "SN", "F10.7obs", "F10.7adj", "D"]
target = "storm_occurred"

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