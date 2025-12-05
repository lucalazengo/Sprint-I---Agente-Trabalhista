#!/bin/bash
set -e

echo "Starting application..."

# Check if indices exist; if not, build them (Self-Healing)
if [ ! -f "data/processed_data/indices/clt_bm25_bge-m3.pkl" ]; then
    echo "Indices not found. Building indices now (this may take a few minutes)..."
    # Ensure dependencies for nltk are ready
    python -c "import nltk; nltk.download('stopwords', download_dir='/opt/nltk_data')"
    python -m src.backend.build_index
else
    echo "Indices found. Skipping build."
fi

# Run the application
echo "Launching Streamlit..."
python -m streamlit run src/frontend/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
