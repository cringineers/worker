import os
import io
import logging
import clip
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv


if not os.path.exists(find_dotenv(".env")):
    logging.warning("Cant find .env file.")
load_dotenv()

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


@app.post("/predict_image")
async def predict_image():
    user = request.form.get('')
    return jsonify(), 200


@app.post("/predict_tag")
async def refresh():
    text = request.form.get("text")
    if text:
        tokens = clip.tokenize(text).to(device)
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.tolist()
        return jsonify(features=text_features), 200
    else:
        return jsonify(error="Can`t find text field"), 400


if __name__ == '__main__':
    logging.warning("Server started")
    app.run(port=5000, host="0.0.0.0")
