from flask import Flask, render_template, request
import run

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
    if request.method == "POST":
        try:
            # Load model only once (could be optimized further)
            model_data = run.load_aqi_model()
            if model_data is None:
                error = "Model file not found on server."
            else:
                model_info, model_pipeline = model_data
                # Get form values
                station_id = int(request.form.get("std-id", 0))
                pm2_5 = float(request.form.get("pm25", 0))
                pm10 = float(request.form.get("pm10", 0))
                no2 = float(request.form.get("no2", 0))
                so2 = float(request.form.get("so2", 0))
                co = float(request.form.get("co", 0))
                o3 = float(request.form.get("o3", 0))
                temp_c = float(request.form.get("temp", 0))
                rain_mm = float(request.form.get("rain", 0))
                # Predict
                result = run.predict_aqi(model_pipeline, station_id, pm2_5, pm10, no2, so2, co, o3, temp_c, rain_mm)
        except Exception as e:
            error = str(e)
    return render_template("advance.html", result=result, error=error)

# Basic route (basic.html)
@app.route("/basic")
def basic():
    return render_template("basic.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)
