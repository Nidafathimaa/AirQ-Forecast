# AirQ-Forecast
The Air Quality Prediction System is a machine learning-based application that predicts AQI (Air Quality Index), PM2.5, and PM10 levels using inputs such as city and date (day, month, year). The project aims to analyze historical air quality data and provide insights into pollution levels for better awareness and decision-making.

The system uses data preprocessing techniques like encoding and scaling to prepare the data, followed by an LSTM (Long Short-Term Memory) model to capture time-based patterns in air quality. The trained model is saved and reused for real-time predictions.

An interactive interface is built using Gradio, allowing users to easily input values and view predictions. The output also includes an AQI category (Good, Moderate, Poor, etc.), making results easy to understand.

This project demonstrates a complete workflow including data preprocessing, model training, prediction, and deployment, showcasing how machine learning can be applied to solve real-world environmental problems.

# Technologies Used
- Python – Core programming
TensorFlow / Keras – Model building (LSTM)
- Pandas & NumPy – Data processing
- Scikit-learn – Data scaling & preprocessing
- Gradio – User interface for interaction

# Dataset Description
The dataset used in this project consists of historical air quality data collected across different cities.
Key Features in Dataset:
- City → Location of measurement
- Date (Day, Month, Year) → Time of observation
- PM2.5 → Fine particulate matter
- PM10 → Coarse particulate matter
- AQI → Air Quality Index (target variable)
Data Characteristics:
- Contains time-based environmental data
- Includes multiple cities
- Used for supervised learning

# Project Workflow
1.Data Preprocessing
 - Handle missing values
 - Encode categorical data (city)
 - Scale input and output features
2.Model Building
 - LSTM (Long Short-Term Memory) neural network
 - Suitable for time-based data
3.Model Training
 - Trained on historical air quality data
 - Optimized using loss function (MSE)
4.Model Saving
 - Saved as .keras file
- Scalers and encoders stored using pickle
5.Prediction System
 - Takes user input (city + date)
 - Applies preprocessing
 - Generates predictions
6.User Interface
 - Built using Gradio
 - Displays results in readable format

# Features of the Application
Predicts:
- AQI (Air Quality Index)
- PM2.5 levels
- PM10 levels
  
Provides AQI category:
- Good 🟢
- Satisfactory 🟡
- Moderate 🟠
- Poor 🔴
- Very Poor 🟣
- Severe ⚫
  
Interactive and user-friendly interface

# Project Structure
MAIN_PROJECTS/
│
├── model.py            # Model training
├── app.py              # User interface (Gradio)
├── air_model.keras     # Trained model
├── scaler_X.pkl        # Input scaler
├── scaler_y.pkl        # Output scaler
├── label.pkl           # Label encoder

# Key Concepts Used
- Machine Learning (Supervised Learning)
- Time Series Modeling (LSTM)
- Data Scaling (MinMaxScaler)
- Model Serialization
- Interactive UI Development

# Future Enhancements
In the future, this project can be improved by integrating real-time air quality data from external APIs to provide live and more accurate predictions. The model performance can be enhanced by using larger and more diverse datasets along with advanced machine learning or deep learning techniques. The application can also be extended to include additional pollutants such as CO, NO₂, and SO₂ for a more comprehensive analysis of air quality. Furthermore, the user interface can be upgraded with interactive visualizations and color-coded indicators to make the results more intuitive. Deploying the application on cloud platforms and enabling location-based automatic input can make the system more accessible and user-friendly for real-world use.
