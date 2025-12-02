# ==========================================
# Stage 1: Builder (Compile dependencies)
# ==========================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Force CPU-only Torch to save ~700MB
# We install to a specific user directory to copy later
RUN pip install --user --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --user --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Runtime (Final Image)
# ==========================================
FROM python:3.11-slim as runtime

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/root/.local/bin:$PATH

WORKDIR /app

# Install runtime system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('stopwords')"

# Copy application code
COPY . .

# Create directory for logs/data ensuring write permissions for appuser
RUN mkdir -p data/processed_data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch
CMD ["python3", "-m", "streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
