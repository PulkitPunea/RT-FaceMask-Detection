# RT-FaceMask-Detection
A Python desktop application for real-time face-mask detection using a trained YOLOv8 model, OpenCV, and a modern GUI built with CustomTkinter. 🔍 Detect masks live via webcam | 🧠 Powered by Ultralytics YOLOv8 | 🎛️ Simple &amp; responsive UI
🧠 Face-Mask Real-Time Detection (YOLOv8 + Tkinter GUI)
A Python desktop application for real-time face mask detection using YOLOv8 and OpenCV, with a modern GUI built using customtkinter.

📸 Features
Real-time camera feed display.

YOLOv8m model integration for detecting face masks.

Clean GUI with start/stop controls.

Mask detection counter.

Real-time FPS overlay.

Maintains aspect ratio with proper padding for consistent display.

Packaged for standalone execution using PyInstaller.

🧰 Tech Stack
Language: Python 3.8+

GUI: customtkinter

Computer Vision: OpenCV

Model: YOLOv8 (ultralytics)

Others: threading, PIL, numpy

📂 Project Structure
css
Copy
Edit
FaceMaskDetection/
│
├── data/
│   ├── best.pt        # Trained YOLOv8 model
│   └── icon.ico       # App icon
├── main.py            # Main application file
├── README.md
└── requirements.txt
🔧 Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Download YOLOv8 weights:

Place your trained model (best.pt) into the data/ directory. You can train a model using Ultralytics YOLO.

🚀 Run the App
bash
Copy
Edit
python main.py
📦 Packaging (Optional)
To convert the script into a standalone executable using PyInstaller:

bash
Copy
Edit
pyinstaller --onefile --windowed --icon=data/icon.ico main.py
Make sure to handle the model and icon using resource_path() to bundle properly with PyInstaller.

🧠 Model Training (optional)
Train your own YOLOv8 model with labeled images for face-mask detection:

bash
Copy
Edit
yolo task=detect mode=train model=yolov8m.pt data=your_data.yaml epochs=50 imgsz=640
🖼️ App Preview
Add screenshots or a GIF showing the app in action here.

📃 License
MIT License

✅ To Do (optional):
 Add logging feature for detection timestamps.

 Add class-wise mask detection display (e.g., mask, no-mask, incorrect-mask).

 Support external camera sources.
