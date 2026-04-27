import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model


model = load_model("air_model.keras")
le = pickle.load(open("label.pkl", "rb"))
scaler_X = pickle.load(open("scaler_X.pkl", "rb"))
scaler_y = pickle.load(open("scaler_y.pkl", "rb"))

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"

def predict(city, day, month, year):
    city_encoded = le.transform([city])[0]

    input_data = np.array([[city_encoded, day, month, year]])
    input_scaled = scaler_X.transform(input_data)
    input_scaled = input_scaled.reshape((1, 1, 4))

    pred = model.predict(input_scaled)
    pred = pred.reshape(1, -1)
    pred = scaler_y.inverse_transform(pred)

    aqi = float(pred[0][2])

    return f"""
AQI: {aqi:.2f}  

PM2.5: {pred[0][0]:.2f}  
PM10: {pred[0][1]:.2f}  

Category: {get_aqi_category(aqi)}
"""

cities = list(le.classes_)

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(list(le.classes_), label="City"),
        gr.Number(label="Day"),
        gr.Number(label="Month"),
        gr.Number(label="Year")
    ],
    outputs=gr.Textbox(lines=6),
    title="Air Quality Prediction",
    flagging_mode="never"   
)

interface.launch()