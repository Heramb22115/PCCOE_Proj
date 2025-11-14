# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# 1. Install System Dependencies (REQUIRED for GeoAlchemy/PostGIS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the NEW backend-only requirements file
COPY requirements.txt .

# 3. Install Python dependencies
# Must install GDAL first to match system library
RUN pip install --no-cache-dir GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')
RUN pip install --no-cache-dir -r backend-requirements.txt

# 4. Copy the application code and model
COPY main.py .
COPY rf_crop_recommendation_model.pkl .

# 5. Expose the port FastAPI will run on
EXPOSE 8000

# 6. Command to run the backend API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]