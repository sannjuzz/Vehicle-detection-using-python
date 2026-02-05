# Vehicle-detection-using-python

A real-time vehicle number plate detection and recognition system using YOLO, EasyOCR, OpenCV, and Tkinter.
The application captures live video from a webcam, detects license plates, extracts text using OCR, and verifies vehicles against a local database through a graphical user interface.

FEATURES

Live webcam feed

YOLO-based license plate detection

EasyOCR for number plate text recognition

Local vehicle database (JSON-based)

Authorized / Unauthorized vehicle identification

Automatic Google Maps route display for authorized vehicles

Admin panel to manage vehicle records

Modern Tkinter graphical user interface

TECH STACK

Python 3.10

OpenCV

EasyOCR

Ultralytics YOLO

PyTorch

Tkinter

NumPy

Pillow

PROJECT STRUCTURE

kerala/
│
├── webcam.py (Main application file)
├── requirements.txt (Required Python libraries)
├── best.pt (YOLO trained model – must be added)
├── vehicle_database.json (Auto-created vehicle database)
└── README.txt / README.md (Project documentation)

SYSTEM REQUIREMENTS

Windows Operating System

Python 3.10.x (64-bit)
NOTE: Python 3.11+ is NOT supported

Webcam

Internet connection (first run only for OCR models)

INSTALLATION

Install Python 3.10
Download Python 3.10.13 (64-bit) from the official Python website.

During installation:

Add Python to PATH

Enable "tcl/tk and IDLE"

Verify installation:
py -3.10 --version

Install required libraries

Open Command Prompt or PowerShell in the project folder and run:

py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install -r requirements.txt

If PyTorch installation fails, install it separately:

py -3.10 -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

Add YOLO model file

Place your trained YOLO model file named:

best.pt

inside the project directory.

RUNNING THE APPLICATION

Navigate to the project folder and run:

py -3.10 webcam.py

ADMIN PANEL

Default admin credentials:

Username: admin
Password: admin123

Admin functionalities:

Add new vehicle records

Set origin and destination

View registered vehicles

WORKING PRINCIPLE

Webcam captures live video

YOLO detects vehicle number plates

EasyOCR extracts text from detected plates

Plate number is normalized and verified

Authorized vehicles open Google Maps route

Unauthorized vehicles are flagged in real time

COMMON ISSUES & SOLUTIONS

ModuleNotFoundError: cv2
→ Install OpenCV using
py -3.10 -m pip install opencv-python

ModuleNotFoundError: tkinter
→ Reinstall Python with "tcl/tk and IDLE" enabled

Camera not opening
→ Close Zoom, Google Meet, or Camera apps

best.pt not found
→ Place YOLO model file in project folder

NOTES

Designed for academic and educational purposes

Uses Python 3.10 only

Tested on Windows OS
