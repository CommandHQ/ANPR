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
- **AWS Lambda Ready**: Includes Mangum integration for serverless deployment.

## Project Structure

```
ANPR/
├── app/
│   ├── main.py              # FastAPI application with API endpoints
│   ├── __init__.py          # Package initialization
│   ├── models/              # Pydantic models for request and response
│   │   ├── request.py       # ImageRequest model
│   │   └── response.py      # OCRResponse model
│   ├── services/            # Business logic
│   │   └── image_processor.py  # ImageProcessor class
│   ├── utils/               # Utility functions
│   │   ├── image_utils.py   # Image processing utilities
│   │   └── logger.py        # Logging configuration
│   └── config/              # Configuration files
├── models/                  # Additional model files
├── last.pt                  # YOLO model file for license plate detection
├── yolov8n.pt              # Additional YOLO model
├── last.onnx               # ONNX version of the model
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── .dockerignore           # Docker ignore file
└── README.md               # This file
```

## Prerequisites

- **Python 3.11+**: Required for running the project.
- **Git**: For cloning the repository.
- **Hardware**: A GPU is recommended for faster YOLO inference with `torch`.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/CommandHQ/ANPR.git
cd ANPR
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

The project requires `ultralytics` (which includes `torch` and `torchvision`), `paddleocr`, `fastapi`, and other dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Verify Model Files

- The YOLO model (`last.pt`) should be in the project root
- Additional models like `yolov8n.pt` and `last.onnx` are also available
- These models are used for license plate detection

## Running the Application

### Start the FastAPI Server

**Important**: The main application file is located in the `app/` directory, so you need to specify the correct module path:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- The API will be available at `http://localhost:8000`
- The `--reload` flag enables auto-reload during development
- Make sure to use `app.main:app` (not just `main:app`)

### Access the API Documentation

- Open your browser and navigate to `http://localhost:8000/docs` to view the interactive Swagger UI
- Alternative documentation: `http://localhost:8000/redoc`
- The Swagger UI provides detailed documentation for the `/ocr` and `/ocr/upload` endpoints

### Expected Startup Warnings

During startup, you may see these warnings (they are normal):

- **PaddleOCR Warning**: About ccache not found - this doesn't affect functionality
- **Ultralytics Warning**: About settings reset - this is normal and doesn't impact performance

## API Endpoints

### 1. `/ocr` (POST)

Process an image to detect and recognize Indian license plates by providing either a base64-encoded image or an image URL.

#### Request Body

```json
{
  "image_base64": "string", // Optional: Base64-encoded image
  "image_url": "string" // Optional: URL of the image
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

### 2. `/ocr/upload` (POST)

Process an uploaded image file to detect and recognize Indian license plates.

#### Example Request

```bash
curl -X POST "http://localhost:8000/ocr/upload" \
  -F "file=@/path/to/license-plate.jpg"
```

#### Response Format (Both Endpoints)

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

## Troubleshooting

### Common Issues

1. **"Could not import module 'main'" Error**

   - Make sure to use `uvicorn app.main:app` instead of `uvicorn main:app`
   - The main.py file is inside the app/ directory

2. **Port Already in Use**

   - Kill existing processes: `lsof -ti:8000 | xargs kill -9`
   - Or use a different port: `--port 8001`

3. **Model Loading Issues**
   - Ensure `last.pt` is in the project root directory
   - Check that the virtual environment is activated

### Performance Notes

- **GPU Acceleration**: YOLO inference with `ultralytics` and `torch` can be resource-intensive. A GPU is recommended for faster processing.
- **Dependencies**: `paddleocr` requires `paddlepaddle`, which is large (~700 MB). Ensure you have sufficient disk space.
- **Memory Usage**: The application loads models into memory at startup, which may take some time initially.

## Docker Support

The project includes Docker configuration files:

```bash
# Build the Docker image
docker build -t anpr-system .

# Run the container
docker run -p 8000:8000 anpr-system
```

## AWS Lambda Deployment

The application includes Mangum integration for serverless deployment on AWS Lambda. The handler is already configured in `app/main.py`.

## License

MIT License. See `LICENSE` for details.

## Contact

For issues or contributions, please open a GitHub issue or contact [kishore@commandhq.com].
