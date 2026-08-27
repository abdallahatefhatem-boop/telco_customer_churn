FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for ONNX and other ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and requirements if they exist (to leverage Docker cache)
COPY pyproject.toml /app/

# Install dependencies (using pip since pyproject.toml has the deps)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy the rest of the application
COPY . /app/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
