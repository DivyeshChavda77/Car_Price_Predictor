from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict_datapoint():

    try:

        data = CustomData(

            km_driven=float(
                request.form.get("km_driven")
            ),

            fuel=request.form.get("fuel"),

            seller_type=request.form.get(
                "seller_type"
            ),

            transmission=request.form.get(
                "transmission"
            ),

            owner=request.form.get(
                "owner"
            ),

            brand=request.form.get(
                "brand"
            ),

            car_age=int(
                request.form.get("car_age")
            )
        )

        pred_df = (
            data.get_data_as_dataframe()
        )

        predict_pipeline = (
            PredictPipeline()
        )

        result = predict_pipeline.predict(
            pred_df
        )

        return render_template(
            "home.html",
            results=round(result[0], 2)
        )

    except Exception as e:

        return render_template(
            "home.html",
            results=str(e)
        )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )