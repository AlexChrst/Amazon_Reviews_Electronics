FROM python:3.11-slim

WORKDIR /app

# Copy the dependency files into the app directory
COPY requirements.txt .

# Install dependencies using uv
RUN pip install -r requirements.txt

# Copy the needded application files in order to the API to run
COPY app.py .

COPY models ./models

# Expose the port
EXPOSE 8000

# Run the FastAPI app via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
