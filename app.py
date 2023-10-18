from flask import Flask, request
from PIL import Image
from transformers import pipeline

app = Flask(__name__)
new_size = 224
classifier = pipeline("image-classification", model="resnet-50-finetuned-eye-disease-2")

def resize_pil_image(image):
    image = Image.open(image)
    aspect_ratio = image.size[0] / image.size[1]
    new_width = int(new_size * aspect_ratio)
    left = (new_width - new_size) / 2
    right = (new_width + new_size) / 2

    return image.resize((new_width, new_size)) \
                .crop((left, 0, right, new_size)) \
                .resize((new_size, new_size))

@app.route("/predict", methods=["POST"])
def predict():
    image = resize_pil_image(request.files['image'])
    return {"predictions": classifier(image)}
