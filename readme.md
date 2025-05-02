Face-Mask Real-Time Detection using YOLOv8 and CustomTkinter
=============================================================

Overview:
---------
This is a Python-based real-time face-mask detection application built using the YOLOv8 object detection model and OpenCV, with a modern graphical user interface (GUI) using customtkinter. The app captures webcam input, detects face masks using a trained YOLOv8 model, and provides real-time feedback through a GUI.

Features:
---------
- Real-time webcam feed display
- Face-mask detection using YOLOv8 (best.pt)
- Modern GUI with CustomTkinter
- Live display of:
  - Detected mask count
  - Current FPS (Frames Per Second)
- Start/Stop buttons for:
  - Camera feed
  - Detection
  - Stopping all processes
- Maintains proper aspect ratio with padded resizing
- Ready for standalone .exe packaging via PyInstaller

Tech Stack:
-----------
- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV
- customtkinter
- PIL (Pillow)
- NumPy
- PyInstaller (for packaging)

Project Structure:
------------------
FaceMaskDetection/
│
├── data/
│   ├── best.pt        # Trained YOLOv8 weights
│   └── icon.ico       # App icon
│
├── main.py            # Main application script
├── README.md          # GitHub readme
└── requirements.txt   # Python dependencies

Setup Instructions:
-------------------
1. Clone the repository:
   git clone https://github.com/PulkitPunea/RT-FaceMask-Detection.git
   cd face-mask-detection

2. (Optional) Create a virtual environment:
   python -m venv venv
   venv\Scripts\activate   (On Windows)
   source venv/bin/activate (On Linux/macOS)

3. Install required dependencies:
   pip install -r requirements.txt

4. Ensure your trained YOLOv8 model (best.pt) is in the 'data/' folder.

5. Run the application:
   python main.py

Training a Custom Model (Optional):
-----------------------------------
You can train your own YOLOv8 model using a dataset with face-mask annotations.

Example command:
yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=50 imgsz=640

Packaging to EXE:
-----------------
To convert the project into an executable:

   pyinstaller --onefile --windowed --icon=data/icon.ico main.py

Make sure your `resource_path()` function is used for loading assets (like model and icon) to support PyInstaller.

License:
--------
This project is open-source and available under the MIT License.

Credits:
--------
- Ultralytics YOLOv8
- Tom Schimansky's CustomTkinter
- OpenCV
- Pillow

Future Improvements:
--------------------
- Add logging/report generation
- Multi-class detection (mask, no-mask, incorrect-mask)
- Support for multiple webcams