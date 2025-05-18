from flask import Flask, render_template, request, Response
import os
import time
import cv2
from werkzeug.utils import secure_filename
import torch
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT

app = Flask(__name__)

# Folder configuration
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# File type restrictions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Fix for PyTorch restricted deserialization
torch.serialization.add_safe_globals([
    Conv,
    DetectionModel
])

# Load the custom YOLO model safely
MODEL_PATH = 'models/yolo11n.pt'
weights = torch.load(MODEL_PATH, map_location='cpu')
model = DetectionModel(cfg=DEFAULT_CFG_DICT['yolov8n'])
model.load_state_dict(weights['model'])
model.eval()

# Utility: Check file type
def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Upload and predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'media' not in request.files:
        return "No file uploaded", 400

    file = request.files['media']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    if ext in ALLOWED_IMAGE_EXTENSIONS:
        return handle_image(upload_path, filename)
    elif ext in ALLOWED_VIDEO_EXTENSIONS:
        return handle_video(upload_path, filename)
    else:
        return "Unsupported file type", 400

# Handle image prediction
def handle_image(path, filename):
    results = model(path)
    annotated_img = results[0].plot()
    pothole_count = len(results[0].boxes) if results[0].boxes is not None else 0

    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    result_img_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    cv2.imwrite(result_img_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    display_path = result_img_path.replace('static/', '')

    return render_template('result.html', media_type='image', media_path=display_path, pothole_count=pothole_count)

# Handle video preview
def handle_video(path, filename):
    return render_template('result.html', media_type='video_stream', video_name=filename)

# Video frame generator
def get_frame(video_path):
    video = cv2.VideoCapture(video_path)
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
        time.sleep(0.03)  # ~30 FPS

# Video feed route
@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(get_frame(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
