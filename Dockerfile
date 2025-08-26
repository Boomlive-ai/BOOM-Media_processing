    # # Use Python 3.11 standard image
    # FROM python:3.11

    # # Set working directory
    # WORKDIR /app

    # # Install system dependencies for OpenCV and other libraries
    # USER root
    # RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

    # # Set environment variables
    # ENV PYTHONDONTWRITEBYTECODE=1 \
    #     PYTHONUNBUFFERED=1 \
    #     PIP_NO_CACHE_DIR=1 \
    #     PIP_DISABLE_PIP_VERSION_CHECK=1

    # # Copy requirements first for better caching
    # COPY requirements.txt .

    # # Install Python dependencies
    # RUN pip install --no-cache-dir -r requirements.txt

    # # Copy application code
    # COPY . .

    # # Create non-root user
    # RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
    # USER app

    # # Expose port
    # EXPOSE 8000

    # # Health check with curl (available in standard python image)
    # HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    #     CMD curl -f http://localhost:8000/health || exit 1

    # # Run the application
    # CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Use Python 3.11 standard image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
USER root
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check with curl (available in standard python image)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]