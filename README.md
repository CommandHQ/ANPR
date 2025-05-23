# LICENSE-PLATE-OCR: Automatic Number Plate Recognition (ANPR) System

## Overview
`LICENSE-PLATE-OCR` is an Automatic Number Plate Recognition (ANPR) system designed to detect and read Indian license plates from images. It uses a YOLO model (via the `ultralytics` library) for license plate detection and `paddleocr` for optical character recognition (OCR) to extract text. The application is built with FastAPI, providing two API endpoints for processing images: one for base64-encoded images or URLs, and another for direct file uploads.

### Features
- **License Plate Detection**: Uses a YOLO model (`last.pt`) with `ultralytics` for efficient detection of Indian license plates.
- **Text Extraction**: Employs `paddleocr` to perform OCR on detected license plates, supporting English text.
- **FastAPI Integration**: Provides two API endpoints:
  - `/ocr`: Accepts base64-encoded images or image URLs.
  - `/ocr/upload`: Accepts direct image file uploads.
- **Modular Design**: Structured for easy integration and local deployment.

## Project Structure
- `app/`: Application code.
  - `models/`: Pydantic models for request and response (`ImageRequest`, `OCRResponse`).
  - `services/`: Image processing logic (`ImageProcessor`).
  - `utils/`: Utility functions (`image_utils`, `logger`).
- `main.py`: FastAPI application with API endpoints and core logic.
- `last.pt`: YOLO model file for license plate detection.
- `.env`: Environment variables (e.g., API keys, if applicable).
- `requirements.txt`: Python dependencies for the project.
- `README.md`: Project documentation.

## Prerequisites
- **Python 3.10**: Required for running the project.
- **Git**: For cloning the repository.
- **Hardware**: A GPU is recommended for faster YOLO inference with `torch`.

## Setup Instructions
### 1. Clone the Repository
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
The project requires `ultralytics` (which includes `torch` and `torchvision`), `paddleocr`, `fastapi`, and other dependencies listed in `requirements.txt`.
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
- Ensure `last.pt` is in the project root or update the path in your `ImageProcessor` class (in `app/services/image_processor.py`) if necessary.

## Running the Application
The `main.py` script sets up a FastAPI application with two endpoints for ANPR.

### 1. Start the FastAPI Server
Run the FastAPI application using `uvicorn`:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- The API will be available at `http://localhost:8000`.
- The `--reload` flag enables auto-reload during development.

### 2. Access the API Documentation
- Open your browser and navigate to `http://localhost:8000/docs` to view the interactive Swagger UI.
- The Swagger UI provides detailed documentation for the `/ocr` and `/ocr/upload` endpoints, including example requests and responses.

## API Endpoints
### 1. `/ocr` (POST)
Process an image to detect and recognize Indian license plates by providing either a base64-encoded image or an image URL.

#### Request Body
```json
{
  "image_base64": "string",  // Optional: Base64-encoded image
  "image_url": "string"      // Optional: URL of the image
}
```
- Provide either `image_base64` or `image_url`, not both.

#### Example Request (Base64)
```bash
curl -X POST "http://localhost:8000/ocr" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "base64-string-here"}'
```

#### Example Request (URL)
```bash
curl -X POST "http://localhost:8000/ocr" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/license-plate.jpg"}'
```

#### Response
```json
{
  "plates": [
    {
      "text": "MH12AB1234",
      "confidence": 0.95,
      "bbox": [100, 150, 300, 200]
    }
  ]
}
```

### 2. `/ocr/upload` (POST)
Process an uploaded image file to detect and recognize Indian license plates.

#### Request
- Upload an image file using `multipart/form-data`.

#### Example Request
```bash
curl -X POST "http://localhost:8000/ocr/upload" \
  -F "file=@/path/to/license-plate.jpg"
```

#### Response
Same as the `/ocr` endpoint:
```json
{
  "plates": [
    {
      "text": "MH12AB1234",
      "confidence": 0.95,
      "bbox": [100, 150, 300, 200]
    }
  ]
}
```

## Notes
- **Performance**: YOLO inference with `ultralytics` and `torch` can be resource-intensive. A GPU is recommended for faster processing.
- **Dependencies**: `paddleocr` requires `paddlepaddle`, which is large (~700 MB). Ensure you have sufficient disk space.
- **Error Handling**: The API includes logging (`app.utils.logger`) and proper error handling for invalid inputs or processing errors.

## License
MIT License. See `LICENSE` for details.

## Contact
For issues or contributions, please open a GitHub issue or contact [kishore@commandhq.com].