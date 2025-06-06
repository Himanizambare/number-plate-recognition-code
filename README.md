﻿# Number Plate Recognition Code
 This project implements an Automatic Number Plate Recognition (ANPR) system using Python, OpenCV, and Tesseract OCR. It processes images to detect and extract vehicle number plates and recognizes the text using Optical Character Recognition.

# 📂 Project Structure
 - main.py: Core script for processing images and extracting number plate text.

 - gui.py: Graphical User Interface for user-friendly interaction.

 - testData/: Directory containing sample images for testing.

 - download.jpeg: Sample image used in the project.

# 🛠️ Features
 - etects number plates from images.

 - Preprocesses images for optimal OCR performance.

 - Extracts and displays recognized text from number plates.

 - Provides a GUI for ease of use.

# 🧰 Requirements

 - Python 3.6 or higher

 - OpenCV

 - NumPy

 - Pillow

 - pytesseract

# 🖥️ Installation

## 1. Clone the repository:

```
git clone https://github.com/Himanizambare/number-plate-recognition-code.git
cd number-plate-recognition-code
```
## 2.Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 3. Install the required packages:

```
pip install -r requirements.txt
```
## 4. Install Tesseract OCR:
### Windows:
Download the installer from [Number Plate Recognition Code](https://github.com/Himanizambare/number-plate-recognition-code)
 and follow the installation instructions.
### Linux:

```
sudo apt-get install tesseract-ocr
```
## Configure pytesseract:

In your Python script, specify the path to the Tesseract executable if it's not in your system's PATH. For example:
```
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```


# 🚀 Usage
## 1.Run the main script:
```
python main.py
```

## 2.Run the GUI:
```
python gui.py
```

# 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

# 🤝 Acknowledgements
 - OpenCV for image processing capabilities.

 - Tesseract OCR for text recognition.

 - Pillow for image handling.


