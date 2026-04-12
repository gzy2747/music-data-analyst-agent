FROM python:3.11-slim

WORKDIR /app

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    matplotlib \
    requests \
    fastapi \
    uvicorn \
    "google-adk>=0.3.0" \
    nest-asyncio \
    python-dotenv \
    pandas \
    numpy \
    scipy \
    python-multipart \
    aiofiles \
    httpx \
    yfinance

# Copy application code
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

CMD ["python", "main.py"]
