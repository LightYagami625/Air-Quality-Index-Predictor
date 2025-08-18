from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    rmse = None
    if request.method == "POST":
        try:
            import Air_Quality_Index_v1
            user_x1 = float(request.form["pm25"])
            user_x2 = float(request.form["pm10"])
            user_x3 = float(request.form["no2"])
            user_x4 = float(request.form["so2"])
            user_x5 = float(request.form["co"])
            user_x6 = float(request.form["o3"])
            file = Air_Quality_Index_v1.pd.read_csv('Air-Pollution.csv')
            x1 = file['PM2.5']
            x2 = file['PM10']
            x3 = file['NO₂']
            x4 = file['SO₂']
            x5 = file['CO']
            x6 = file['O₃']
            y1 = file['AQI_Target']
            b = Air_Quality_Index_v1.multiple_linear_regression(x1, x2, x3, x4, x5, x6, y1)
            y_predict = Air_Quality_Index_v1.prediction(b, user_x1, user_x2, user_x3, user_x4, user_x5, user_x6)
            rmse = Air_Quality_Index_v1.np.sqrt(Air_Quality_Index_v1.mean_squared_error([Air_Quality_Index_v1.np.mean(y1)], [y_predict]))
            result = f"{y_predict:.4f}"
            rmse = f"{rmse:.4f}"
        except Exception as e:
            result = f"Error: {e}"
    return render_template("index.html", result=result, rmse=rmse)

if __name__ == "__main__":
    app.run(debug=True)
