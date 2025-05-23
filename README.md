# LICENSE-PLATE-OCR: Automatic Number Plate Recognition (ANPR) System

## Overview
`LICENSE-PLATE-OCR` is an Automatic Number Plate Recognition (ANPR) system designed to detect and read license plates from images. It uses a YOLO model (via the `ultralytics` library) for license plate detection and `paddleocr` for optical character recognition (OCR) to extract the text. This project is intended for local deployment and can be extended for various environments.

### Features
- **License Plate Detection**: Uses a YOLO model (`last.pt`) with `ultralytics` for efficient detection.
- **Text Extraction**: Employs `paddleocr` to perform OCR on detected license plates, supporting English text.
- **Modular Design**: Structured for easy integration into different applications (e.g., web apps, scripts).

## Project Structure
- `app/`: Application code (e.g., API endpoints, utility functions).
- `models/`: Model-related code (if any additional preprocessing/postprocessing logic exists).
- `main.py`: Core logic for YOLO inference and OCR.
- `last.pt`: YOLO model file for license plate detection.
- `.env`: Environment variables (e.g., API keys, if applicable).
- `requirements.txt`: Python dependencies for the project.

## Prerequisites
- **Python 3.10**: Required for running the project.
- **Git**: For cloning the repository.
- **Hardware**: A GPU is recommended for faster YOLO inference with `torch`.

## Setup Instructions
### 1. Clone the Repositorya
```bash
git clone https:git@github.com-commandhq:CommandHQ/ANPR.git
cd license-plate-ocr
```

### 2. Set Up a Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
The project requires `ultralytics` (which includes `torch` and `torchvision`), `paddleocr`, and other dependencies listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

Example `requirements.txt`:
```
ultralytics==8.2.28
paddleocr==2.10.0
opencv-python-headless==4.8.1.78
fastapi==0.103.2
uvicorn==0.24.0
mangum==0.17.0
pydantic==1.10.13
python-multipart==0.0.6
requests==2.31.0
python-dotenv==0.21.1
```

### 4. Prepare the YOLO Model
- The YOLO model (`last.pt`) is included in the repository and used for license plate detection.
- Ensure `last.pt` is in the project root or update the path in `main.py` if necessary.

## Usage
### Running the Application
The `main.py` script provides a basic example of using YOLO for detection and `paddleocr` for OCR.

1. **Update `main.py` (if needed)**:
   Ensure the paths to `last.pt` and your image files are correct.

   Example `main.py`:
   ```python
   from ultralytics import YOLO
   from paddleocr import PaddleOCR
   import cv2

   def run_yolo_inference(image_path):
       model = YOLO("last.pt")
       results = model.predict(image_path)
       return results

   def run_ocr(image_path):
       ocr = PaddleOCR(use_angle_cls=True, lang='en')
       result = ocr.ocr(image_path, cls=True)
       return result

   if __name__ == "__main__":
       image_path = "sample.jpg"
       detections = run_yolo_inference(image_path)
       ocr_result = run_ocr(image_path)
       print("Detections:", detections)
       print("OCR Result:", ocr_result)
   ```

2. **Run the Script**:
   ```bash
   python main.py
   ```

### Example Output
- **Input**: An image (`sample.jpg`) containing a license plate.
- **Output**:
  - YOLO detections: Bounding boxes of detected license plates.
  - OCR result: Extracted text (e.g., "ABC123").

## Notes
- **Performance**: YOLO inference with `ultralytics` and `torch` can be resource-intensive. A GPU is recommended for faster processing.
- **Dependencies**: `paddleocr` requires `paddlepaddle`, which is large (~700 MB). Ensure you have sufficient disk space.
- **Customization**: Modify `main.py` to integrate with a web API (e.g., using `fastapi`) or other workflows.

## License
MIT License. See `LICENSE` for details.

## Contact
For issues or contributions, please open a GitHub issue or contact [kishore@commandhq.com].