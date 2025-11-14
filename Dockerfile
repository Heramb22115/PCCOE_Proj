# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# 1. Install System Dependencies (REQUIRED for GeoAlchemy/PostGIS)
# libpq-dev is for psycopg (Postgres)
# gdal-bin and libgdal-dev are for GeoAlchemy2
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements and install Python dependencies
COPY requirements.txt .
# Install GDAL first, then the rest
RUN pip install --no-cache-dir GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the application code
# (This copies main.py, app.py, .pkl file, etc.)
COPY . .

# 4. Create a startup script to run both Backend and Frontend
RUN echo '#!/bin/bash\n\
echo "Starting FastAPI Backend on port 8000..."\n\
uvicorn main:app --host 0.0.0.0 --port 8000 & \n\
echo "Starting Streamlit Frontend on port 8501..."\n\
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false \n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose ports for Streamlit (UI) and FastAPI (Backend)
EXPOSE 8501
EXPOSE 8000

# Run the startup script
CMD ["/app/start.sh"]