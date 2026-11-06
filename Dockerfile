# Dockerfile — Reproducible Environment for ASD Screening Research Platform
# Base: Python 3.11 slim
FROM python:3.11-slim

# Avoid prompts from debian frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install essential system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit plotly

# Copy project files
COPY . /app/

# Expose ports: 8501 for Streamlit demo, 8000 for static webapp
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch Streamlit demonstration platform
CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
