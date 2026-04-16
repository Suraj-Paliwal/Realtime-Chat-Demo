FROM python:3.11.9-slim

WORKDIR /app

# Install system dependencies needed by soundfile, numba, etc.
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8765

CMD ["python", "bot.py"]
