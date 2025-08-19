from flask import Flask, render_template, request
import run

# Global variables to store model data
model_info = None
model_pipeline = None

def load_model():
    """Load the model once at startup"""
    global model_info, model_pipeline
    print("Loading model at startup...")
    model_data = run.load_aqi_model()
    if model_data is not None:
        model_info, model_pipeline = model_data
        print("✅ Model loaded successfully!")
        return True
    print("❌ Failed to load model!")
    return False

def get_aqi_category(aqi):
    """Convert AQI value to category and color"""
    if aqi <= 50:
        return "Good ", "#00E400"
    elif aqi <= 100:
        return "Satisfactory ", "#FFFF00"
    elif aqi <= 200:
        return "Moderate ", "#FF7E00"
    elif aqi <= 300:
        return "Poor ", "#FF0000"
    elif aqi <= 400:
        return "Very Poor ", "#8F3F97"
    else:
        return "Severe ", "#7E0023"

# Use standard folders: templates/ and static/
app = Flask(__name__, template_folder='template', static_folder='static')

# Load model at startup
if not load_model():
    print("⚠️ Warning: Application started without model!")


# Home route (index.html)
@app.route("/")
def index():
    return render_template("index.html")

# Advance prediction route
@app.route("/advance", methods=["GET", "POST"])
def advance():
    global model_pipeline
    result = None
    error = None
    prediction_data = None
    
    if request.method == "POST":
        try:
            if model_pipeline is None:
                if not load_model():  # Try loading model if it's not loaded
                    error = "Model is not available. Please try again later."
                    return render_template("advance.html", error=error)
            
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
            
            # Make prediction using the global model
            prediction = run.predict_aqi(model_pipeline, **input_values)
            print(f"Prediction result: {prediction}")  # Debug log
            
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
    app.run(debug=True, host="0.0.0.0", port=8000)