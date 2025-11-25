# Use an official Python runtime as a parent image
# Slim version to reduce size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# Note: We force CPU-only torch to save space (critical for cloud deployments)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (Stopwords)
RUN python -c "import nltk; nltk.download('stopwords')"

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the app
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

