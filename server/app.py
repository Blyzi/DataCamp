from flask import Flask, request
from flask_cors import CORS
from PIL import Image
import torch
from torchvision.transforms import v2 as transforms
from models.classifier import MultiLabelImageClassifierModel, normalize_image, get_image_variations, resize_image

app = Flask(__name__)
CORS(app)

# LABELS = ['Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'LS', 'CSR', 'ODC', 'CRVO', 'ODP', 'ODE', 'RS', 'CRS', 'RPEC']
LABELS = ['Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ODC', 'ODP', 'ODE']
IMAGE_SIZE = 250
NUM_CHANNELS = 7
TEST_SCORE_THRESHOLD = 0.1

model = MultiLabelImageClassifierModel(num_classes=len(LABELS), input_size=IMAGE_SIZE, num_channels=NUM_CHANNELS)
model.load_state_dict(torch.load(f'./models/Hope.pth'))
model.eval()

def prepare_image(image):
    image = Image.open(image)

    image = resize_image(image)

    image = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])(image)

    image = transforms.ToPILImage()(image)
    image_layers = get_image_variations(image)

    for i, img in enumerate(image_layers):
        image_layers[i] = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])(img)

    image_layers[0] = normalize_image(image_layers[0])

    image = torch.cat(image_layers, dim=0)

    image = image.unsqueeze(0)

    return image

@app.route("/predict", methods=["POST"])
def predict():
    image = prepare_image(request.files['image'])

    predictions = model(image)
    predictions = sorted(enumerate(predictions[0]), key=lambda x: x[1], reverse=True)
    predictions = [(LABELS[label_idx], score) for label_idx, score in predictions if score > TEST_SCORE_THRESHOLD]

    return {
        "predictions": [
            {"label": label, "score": float(score)} for label, score in predictions
        ]
    }
