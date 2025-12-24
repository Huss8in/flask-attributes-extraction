# GradProject Dockerfile - GPU-enabled Flask service for image attribute extraction
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY grad_app.py .

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python3", "grad_app.py"]
