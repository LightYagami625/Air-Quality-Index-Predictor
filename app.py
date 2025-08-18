from flask import Flask, render_template, request
import run


app = Flask(__name__, template_folder='template')


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
            # Load model only once (could be optimized)
            model_data = run.load_aqi_model()
            if model_data is None:
                error = "Model not found."
            else:
                model_info, model_pipeline = model_data
                # Get form values
                station_id = int(request.form["std-id"])
                pm2_5 = float(request.form["pm25"])
                pm10 = float(request.form["pm10"])
                no2 = float(request.form["no2"])
                so2 = float(request.form["so2"])
                co = float(request.form["co"])
                o3 = float(request.form["o3"])
                temp_c = float(request.form["temp"])
                rain_mm = float(request.form["rain"])
                # Predict
                result = run.predict_aqi(model_pipeline, station_id, pm2_5, pm10, no2, so2, co, o3, temp_c, rain_mm)
        except Exception as e:
            error = str(e)
    return render_template("advance.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
