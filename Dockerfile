# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user early
RUN useradd --create-home --shell /bin/bash anpr

# Create PaddleOCR directory structure and download models
RUN mkdir -p /home/anpr/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer && \
    mkdir -p /home/anpr/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer && \
    mkdir -p /home/anpr/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer

# Download and keep PaddleOCR models (both tar and extracted for optimal performance)
RUN cd /home/anpr/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer && \
    wget -O en_PP-OCRv3_det_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && \
    tar -xf en_PP-OCRv3_det_infer.tar

RUN cd /home/anpr/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer && \
    wget -O en_PP-OCRv4_rec_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar && \
    tar -xf en_PP-OCRv4_rec_infer.tar

RUN cd /home/anpr/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer && \
    wget -O ch_ppocr_mobile_v2.0_cls_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar && \
    tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar

# Copy application code
COPY app/ ./app/
COPY *.pt ./
COPY *.onnx ./

# Set ownership of all files to anpr user
RUN chown -R anpr:anpr /app /home/anpr

# Switch to non-root user
USER anpr

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 