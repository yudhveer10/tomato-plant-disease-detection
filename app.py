from flask import Flask, render_template,request,jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np

model=load_model("model.keras")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

class_labels = [
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf_Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if request.method == "POST":
        if 'image' not in request.files:
            print("No image part")
            return jsonify({"error": "No image part"})

        
        image_file = request.files["image"]
        if image_file.filename == '':
            print("No selected file")
            return jsonify({"error": "No selected file"})
    
        image_path = os.path.join("image.jpg")
        image_file.save(image_path)

        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  
        image_array = image_array / 255.0

        pred=model.predict(image_array)
        predicted_class_index = np.argmax(pred, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        print(predicted_class_label)
        return jsonify({"result": predicted_class_label})

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=3000)
