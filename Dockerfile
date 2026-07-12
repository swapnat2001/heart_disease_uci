# Use a stable, lightweight Python base image
FROM python:3.11-slim

# Set system-level environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Establish execution workspace path
WORKDIR /app

# Install system dependencies if required, then inject requirements matrix
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the core project structure over to the container filesystem
COPY . .

# Run data integrity and constraint validation assertions as a build gate
RUN python -m pytest src/test_data.py || python -m unittest discover -s src/ -p "test_*.py"

# Execute training pipeline to produce the serialization binary (heart_disease_pipeline.joblib)
RUN python src/train.py

# Expose the production microservice application port
EXPOSE 8000

# Fire up the asynchronous production engine
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]