Got it! I'll update the README to mention the sample images you added:

````markdown
# Pothole Detection Web App

![Pothole Detection](https://i.imgur.com/your-image-link.png)  
*A web application for detecting potholes in images and videos using YOLO deep learning model.*

---

## Overview

This project is a Flask-based web application that detects potholes from uploaded images and videos, as well as live video feeds via webcam. It uses the **YOLO (You Only Look Once)** object detection model for real-time and batch pothole detection.

---

## Features

- Upload images (`png`, `jpg`, `jpeg`) or videos (`mp4`, `avi`, `mov`) for pothole detection.
- View annotated results showing detected potholes with bounding boxes.
- Detect potholes in live webcam video streams.
- Responsive and clean UI with Tailwind CSS and modern design.
- Backend built with Flask and YOLOv8 model from Ultralytics.

---

## Demo

- Upload an image or video, then see the potholes detected and highlighted.
- Stream live video from your webcam with real-time pothole detection.

---

## Installation

### Prerequisites

- Python 3.8+
- `pip` package manager

### Clone the repo

```bash
git clone https://github.com/nvakumar/pothole_detection.git
cd pothole_detection
````

### Install dependencies

```bash
pip install -r requirements.txt
```

Make sure to have:

* `flask`
* `torch`
* `opencv-python`
* `ultralytics`

### Download YOLO model

Place your trained YOLO model weights file `yolo11n.pt` inside the `models/` directory.

---

## Usage

### Run the Flask app

```bash
python app.py
```

By default, it runs on `http://0.0.0.0:5000/`.

### Features

* Go to the home page and upload an image or video.
* View pothole detection results on a new page.
* For video uploads, results are streamed frame by frame with bounding boxes.
* For live detection via webcam, click the “detect via camera” link.

---

## File Structure

```
pothole_detection/
│
├── app.py                    # Flask app with detection logic
├── requirements.txt          # Python dependencies
├── models/
│   └── yolo11n.pt            # YOLO trained weights file
├── static/
│   ├── images/               # Original uploaded images
│   │    └── sample1.png     # Example original image
│   ├── results/              # Annotated predicted images
│   │    ├── predict1.png     # Example prediction image 1
│   │    └── predict2.png     # Example prediction image 2
│   └── css/                  # (Optional) CSS files
├── templates/
│   ├── index.html            # Home page upload form
│   └── result.html           # Detection results page
└── README.md
```

---

## How it Works

* User uploads media file.
* Flask saves the file locally in the `static/images/` folder.
* YOLO model processes the media to detect potholes.
* Annotated images are saved in the `static/results/` folder (e.g., `predict1.png`, `predict2.png`).
* Annotated images or live video streams are served back to the user.
* For videos, frames are extracted and detection results streamed in near real-time.

---

## Technologies Used

* Python 3.8+
* Flask (Web framework)
* PyTorch (Deep learning backend)
* Ultralytics YOLOv8 model
* OpenCV (Image and video processing)
* Tailwind CSS (Frontend styling)

---

## Contact

**Ch N V Ajay Kumar**
Email: [nvakumarch@gmail.com](mailto:nvakumarch@gmail.com)
GitHub: [https://github.com/nvakumar/pothole\_detection](https://github.com/nvakumar/pothole_detection)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* OpenCV community
* Flask community

---

Feel free to contribute or raise issues for any improvements or bugs!

```

---

This update clearly explains where your original and predicted images are stored, referencing your sample files `predict1.png` and `predict2.png` in the results folder, and original images in the images folder. Would you like me to help you with the `requirements.txt` or any setup instructions next?
```
