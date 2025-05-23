# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for memory optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PADDLEOCR_HOME=/home/anpr/.paddleocr

# Install system dependencies needed by Ultralytics and OpenCV
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with no cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user early
RUN useradd --create-home --shell /bin/bash anpr

# Create PaddleOCR directory structure
RUN mkdir -p /home/anpr/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer && \
    mkdir -p /home/anpr/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer && \
    mkdir -p /home/anpr/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer

# Download and extract PaddleOCR models to the correct location
# For detection model
RUN cd /home/anpr/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer && \
    wget -O en_PP-OCRv3_det_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && \
    tar -xf en_PP-OCRv3_det_infer.tar && \
    mv en_PP-OCRv3_det_infer/* . && \
    rm -rf en_PP-OCRv3_det_infer

# For recognition model
RUN cd /home/anpr/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer && \
    wget -O en_PP-OCRv4_rec_infer.tar https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar && \
    tar -xf en_PP-OCRv4_rec_infer.tar && \
    mv en_PP-OCRv4_rec_infer/* . && \
    rm -rf en_PP-OCRv4_rec_infer

# For classification model
RUN cd /home/anpr/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer && \
    wget -O ch_ppocr_mobile_v2.0_cls_infer.tar https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar && \
    tar -xf ch_ppocr_mobile_v2.0_cls_infer.tar && \
    mv ch_ppocr_mobile_v2.0_cls_infer/* . && \
    rm -rf ch_ppocr_mobile_v2.0_cls_infer

# Copy application code (only necessary files)
COPY app/ ./app/
# Use the custom trained model
COPY last.pt ./

# Set ownership of all files to anpr user
RUN chown -R anpr:anpr /app /home/anpr

# Switch to non-root user
USER anpr

# Expose port
EXPOSE 8000

# Health check with reduced interval
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Command to run the application with memory optimization
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 