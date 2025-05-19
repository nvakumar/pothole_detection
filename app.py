from flask import Flask, render_template, request, Response, url_for
import os
import time
import cv2
import torch
from pathlib import Path
from torch.nn import Sequential, ModuleList, Conv2d
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f
from werkzeug.utils import secure_filename

# ✅ Allowlist classes for safe deserialization
torch.serialization.add_safe_globals([
    DetectionModel,
    Conv,
    C2f,
    Sequential,
    ModuleList,
    Conv2d
])

app = Flask(__name__)

# Folder configuration
UPLOAD_FOLDER = Path('static/uploads')
RESULT_FOLDER = Path('static/results')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['RESULT_FOLDER'] = str(RESULT_FOLDER)

# File type restrictions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

# ✅ Load YOLO model
MODEL_PATH = 'models/yolo11n.pt'
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    raise e


def allowed_file(filename, allowed_set):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'media' not in request.files:
        return "No file uploaded", 400

    file = request.files['media']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_FOLDER / filename
    file.save(str(upload_path))

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return handle_image(upload_path, filename)
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        return handle_video(upload_path, filename)
    else:
        return "Unsupported file type", 400


def handle_image(image_path, filename):
    """Handle image input and return results."""
    results = model(str(image_path))
    annotated_img = results[0].plot()
    pothole_count = len(results[0].boxes) if results[0].boxes is not None else 0

    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
    result_img_path = RESULT_FOLDER / filename
    cv2.imwrite(str(result_img_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    return render_template('result.html',
                           media_type='image',
                           media_path=f'results/{filename}',
                           pothole_count=pothole_count)


def handle_video(video_path, filename):
    """Render video page with streaming endpoint."""
    return render_template('result.html',
                           media_type='video_stream',
                           video_name=filename)


def get_frame(video_path):
    """Generator for video frame streaming with annotations."""
    video = cv2.VideoCapture(str(video_path))
    try:
        while True:
            success, frame = video.read()
            if not success:
                break
            results = model(frame[..., ::-1])
            annotated = results[0].plot()
            ret, jpeg = cv2.imencode('.jpg', annotated)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.03)  # approx. 30 FPS
    finally:
        video.release()


@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = UPLOAD_FOLDER / filename
    return Response(get_frame(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
