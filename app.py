from flask import Flask, render_template, request, send_file
import torch
import cv2
import os
from PIL import Image
from torchvision.transforms.functional import to_tensor
from model.model import MattingNetwork  # from RVM repo

app = Flask(__name__)

# Load model
model = MattingNetwork("mobilenetv3")
model.load_state_dict(torch.load("rvm_mobilenetv3.pth", map_location="cpu"))
model = model.eval()

def remove_background(video_path, output_path="static/output.mp4"):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        src_tensor = to_tensor(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            fgr, pha, *_ = model(src_tensor)
        pha_np = pha.squeeze().cpu().numpy()
        fgr_np = fgr.squeeze().permute(1, 2, 0).cpu().numpy()

        comp = fgr_np * pha_np[..., None] + (1 - pha_np[..., None]) * 1
        comp = (comp * 255).astype("uint8")
        frames.append(comp)

    cap.release()
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]
        input_path = "input.mp4"
        video.save(input_path)

        remove_background(input_path)
        return render_template("index.html", processed=True)

    return render_template("index.html", processed=False)

@app.route("/download")
def download():
    return send_file("static/output.mp4", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
