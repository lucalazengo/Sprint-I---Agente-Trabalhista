# ==========================================
# Stage 1: Builder (Compile dependencies)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Force CPU-only Torch to save ~700MB
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data to a specific directory
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/opt/nltk_data')"

# ==========================================
# Stage 2: Runtime (Final Image)
# ==========================================
FROM python:3.11-slim AS runtime

# Security: Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    NLTK_DATA="/opt/nltk_data"

WORKDIR /app

# Install runtime system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and NLTK data from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nltk_data /opt/nltk_data

# Copy application code
COPY . .

# Copy entrypoint script and make executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create directory for logs/data ensuring write permissions for appuser
RUN mkdir -p data/processed_data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch with entrypoint for self-healing indices
CMD ["./entrypoint.sh"]
