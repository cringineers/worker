import io
import logging
import clip
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
torch.set_grad_enabled(False)


@app.post("/predict_image")
async def predict_image():
    try:
        body = request.get_json()
        images = body.get("images")
        tag_groups = body.get("tag_groups")
        if tag_groups and images:
            image_ls = torch.tensor([image["latent_space"] for image in images], dtype=torch.float32, device=device)
            result = {}
            for group in tag_groups:
                group_id, group_name, tags = group["id"], group["name"], group["tags"]
                tags_spaces = [tag["latent_space"] for tag in tags]
                tags_spaces = torch.tensor(tags_spaces, dtype=torch.float32, device=device)
                prediction = (100.0 * image_ls @ tags_spaces.T).argmax(dim=1)
                result[group_id] = [{"image_id": images[i]["id"], "tag_id": tags[pred.item()]["id"]} for i, pred in enumerate(prediction)]
            return jsonify(result), 200
        return jsonify(error="Can`t find image and tag_groups fields"), 400
    except Exception as err:
        return jsonify(error="Unknown formats"), 400


@app.post("/features_image")
async def features_image():
    try:
        if request.data:
            buffer = io.BytesIO(request.data)
            image = Image.open(buffer)
            image = torch.stack([preprocess(image)]).to(device)
            features = model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.tolist()[0]
            return jsonify(features=features), 200
        return jsonify(error="Can`t find image"), 400
    except Exception as err:
        return jsonify(error="Unknown image format"), 400


@app.post("/features_tag")
async def features_tag():
    body = request.get_json()
    text = body.get("text")
    if text:
        tokens = clip.tokenize(text).to(device)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.tolist()[0]
        return jsonify(features=text_features), 200
    return jsonify(error="Can`t find text field"), 400


if __name__ == '__main__':
    logging.warning("Server started")
    app.run(port=5000, host="0.0.0.0")
