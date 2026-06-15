import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv('ev_data.csv')

X = df[['battery_pct', 'speed_kmh', 'temperature_c', 'passengers']]
y = df['range_km']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"Mean Absolute Error : {mae:.2f} km")
print(f"R2 Score            : {r2:.4f}  (1.0 = perfect)")

joblib.dump(model, 'ev_model.pkl')
print("Model saved as ev_model.pkl")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue', edgecolors='none')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Range (km)')
plt.ylabel('Predicted Range (km)')
plt.title('Actual vs Predicted EV Range')
plt.legend()
plt.tight_layout()
plt.savefig('model_accuracy.png')
print("Chart saved as model_accuracy.png")
