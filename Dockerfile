FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with optimization
RUN pip install --no-cache-dir --compile -r requirements.txt

# Copy application
COPY app.py .
COPY templates/ ./templates/
COPY *.pkl ./ 2>/dev/null || true
COPY *.csv ./ 2>/dev/null || true

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

CMD ["python", "app.py"]
