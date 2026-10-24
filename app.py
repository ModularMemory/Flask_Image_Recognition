"""Defines the main webserver logic for the prediction application."""

# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    """
    Renders index.html as the default route.
    """

    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """
    Predicts the digit hand symbol in the uploaded file and renders the
    prediction result to result.html.

    Renders result.html with custom error messages on failure.
    """

    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))

        return render_template("result.html", err="Illegal request method.")

    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
