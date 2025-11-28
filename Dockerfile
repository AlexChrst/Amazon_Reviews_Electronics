FROM python:3.11-slim

# Set the working directory, if it doesn't exist, create it
WORKDIR /app

# Update the Debian package list and install curl and ca-certificates in order to download the uv tool
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Run the command to install uv.  "| sh" send the downloaded script to the shell for immediate execution 
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add the local user binary directory to the PATH environment variable
ENV PATH="/root/.local/bin:$PATH"

# Copy the dependency files into the app directory
COPY pyproject.toml uv.lock ./

# Install the dependencies in a virtual environment with uv
RUN uv sync --frozen --no-install-project

# Copy the API code into the container
COPY app.py .

# Copy the models directory into the container
COPY models ./models

ENV PATH="/app/.venv/bin:$PATH"

# Expose the port
EXPOSE 8000

# Run the FastAPI app via uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
