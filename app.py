import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="⚡",
    layout="centered"
)

model = joblib.load('ev_model.pkl')

st.title("⚡ EV Range Predictor")
st.caption("Built by Zain Abbas · Mechanical Engineering + AI")
st.markdown("Adjust the inputs below to predict how far your EV can travel.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    battery_pct = st.slider(
        "🔋 Battery level (%)",
        min_value=10, max_value=100, value=80, step=1
    )
    speed_kmh = st.slider(
        "🚗 Average speed (km/h)",
        min_value=20, max_value=130, value=60, step=1
    )

with col2:
    temperature_c = st.slider(
        "🌡 Outside temperature (°C)",
        min_value=-5, max_value=45, value=22, step=1
    )
    passengers = st.slider(
        "👥 Number of passengers",
        min_value=1, max_value=5, value=2, step=1
    )

st.divider()

input_data = np.array([[battery_pct, speed_kmh, temperature_c, passengers]])
predicted_range = model.predict(input_data)[0]
predicted_range = max(0, round(predicted_range, 1))

st.subheader("Predicted range")

m1, m2, m3 = st.columns(3)
m1.metric("Estimated range", f"{predicted_range} km")
m2.metric("Battery level",   f"{battery_pct}%")
m3.metric("At speed",        f"{speed_kmh} km/h")

if predicted_range > 200:
    st.success(f"Great range! Your EV can travel approximately {predicted_range} km.")
elif predicted_range > 100:
    st.warning(f"Moderate range. Your EV can travel approximately {predicted_range} km.")
else:
    st.error(f"Low range. Consider charging before your trip. (~{predicted_range} km left)")

st.divider()

st.subheader("Range vs battery level")
st.caption("How predicted range changes as battery drains (at current speed & conditions)")

battery_levels = list(range(10, 101, 5))
ranges = []
for b in battery_levels:
    r = model.predict([[b, speed_kmh, temperature_c, passengers]])[0]
    ranges.append(max(0, round(r, 1)))

chart_df = pd.DataFrame({
    'Battery (%)': battery_levels,
    'Range (km)':  ranges
})
st.line_chart(chart_df.set_index('Battery (%)'))

st.divider()
st.subheader("What affects your range?")

impact_data = {
    'Factor': ['Battery %', 'Speed', 'Temperature', 'Passengers'],
    'Your value': [
        f"{battery_pct}%",
        f"{speed_kmh} km/h",
        f"{temperature_c}°C",
        str(passengers)
    ],
    'Impact on range': [
        'Higher = more range',
        'Higher speed = less range',
        'Extreme cold/heat = less range',
        'More passengers = less range'
    ]
}
st.table(pd.DataFrame(impact_data))

st.caption("Model: Gradient Boosting Regressor · R² = 0.98 · Built with Python & Streamlit")
