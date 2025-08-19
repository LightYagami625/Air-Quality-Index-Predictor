from flask import Flask, render_template, request
import run

def get_aqi_category(aqi):
    """Convert AQI value to category and color"""
    if aqi <= 50:
        return "Good 🟢", "#00E400"
    elif aqi <= 100:
        return "Satisfactory 🟡", "#FFFF00"
    elif aqi <= 200:
        return "Moderate 🟠", "#FF7E00"
    elif aqi <= 300:
        return "Poor 🔴", "#FF0000"
    elif aqi <= 400:
        return "Very Poor 🟣", "#8F3F97"
    else:
        return "Severe ⚫", "#7E0023"

# Use standard folders: templates/ and static/
app = Flask(__name__, template_folder='template', static_folder='static')


# Home route (index.html)
@app.route("/")
def index():
    return render_template("index.html")

# Advance prediction route
@app.route("/advance", methods=["GET", "POST"])
def advance():
    result = None
    error = None
    prediction_data = None
    if request.method == "POST":
        try:
            print("Loading model...")  # Debug log
            model_data = run.load_aqi_model()
            if model_data is None:
                error = "Model file not found on server."
            else:
                model_info, model_pipeline = model_data
                print("Model loaded successfully")  # Debug log
                
                # Get form values
                input_values = {
                    'station_id': int(request.form.get("std-id", 0)),
                    'pm2_5': float(request.form.get("pm25", 0)),
                    'pm10': float(request.form.get("pm10", 0)),
                    'no2': float(request.form.get("no2", 0)),
                    'so2': float(request.form.get("so2", 0)),
                    'co': float(request.form.get("co", 0)),
                    'o3': float(request.form.get("o3", 0)),
                    'temp_c': float(request.form.get("temp", 0)),
                    'rain_mm': float(request.form.get("rain", 0))
                }
                print(f"Form data received: {input_values}")  # Debug log
                
                # Make prediction
                prediction = run.predict_aqi(model_pipeline, **input_values)
                print(f"Prediction result: {prediction}")  # Debug log
                run.get_aqi_category(prediction)
                
                if prediction['success']:
                    result = prediction['predicted_aqi']
                    prediction_data = prediction
                else:
                    error = prediction['error']
                    
        except Exception as e:
            print(f"Error occurred: {str(e)}")  # Debug log
            error = str(e)
    return render_template("advance.html", 
                           result=result,
                           error=error,
                           prediction=prediction_data)

# Basic route (basic.html)
@app.route("/basic")
def basic():
    return render_template("basic.html")

@app.route("/info")
def info():
    return render_template("info.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8880)
