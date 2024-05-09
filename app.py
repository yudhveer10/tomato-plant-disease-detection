from flask import Flask, render_template
import model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict_image", method=["POST"])
def predict_image():
    # yaha prediction wala bna lena
    prediction = "l"

    return prediction

app.run(debug=True, port=3000)
