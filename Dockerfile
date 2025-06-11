FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install pytest for testing
RUN pip install pytest pytest-asyncio

# Copy the source code
COPY src/ src/
COPY tests/ tests/
COPY streamlit_app.py .
COPY pytest.ini .

# Set PYTHONPATH
ENV PYTHONPATH=/app/src:/app

# Create downloads directory
RUN mkdir -p downloads test_downloads

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]