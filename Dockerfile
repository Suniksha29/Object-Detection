FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TORCH_NUM_THREADS=2 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Set working directory
WORKDIR /app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache model weights into image layer so runtime inference is instant without network download
RUN python -c "from transformers import RTDetrImageProcessor, RTDetrForObjectDetection; RTDetrImageProcessor.from_pretrained('PekingU/rtdetr_r50vd'); RTDetrForObjectDetection.from_pretrained('PekingU/rtdetr_r50vd', low_cpu_mem_usage=True)"

# Copy application source code
COPY app/ ./app/
COPY models/ ./models/
COPY static/ ./static/
COPY run.py .

# Expose server ports (8000 standard, 10000 Render default)
EXPOSE 8000
EXPOSE 10000

# Launch server directly via Python CLI entrypoint (respects $PORT and $HOST)
CMD ["python", "run.py"]
