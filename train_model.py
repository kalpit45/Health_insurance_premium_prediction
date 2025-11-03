import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("C:\Users\Kalpit Sharma\Desktop\Health_Insurance_Premium_Prediction\data\Health_insurance.csv")

# Preprocessing
data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
data["region"] = data["region"].map({"southwest": 0, "southeast": 1,
                                     "northwest": 2, "northeast": 3})

# Features and label
X = data[["age", "sex", "bmi", "smoker"]]
y = data["charges"]

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=300, max_depth=15)
model.fit(xtrain, ytrain)

# Predictions
ypred = model.predict(xtest)

# Evaluation
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ypred)

print("✅ Model Evaluation")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
joblib.dump(model, "C:\Users\Kalpit Sharma\Desktop\Health_Insurance_Premium_Prediction\app\insurance_model.pkl")
print("✅ Model saved to app/insurance_model.pkl")
