FROM python:3.12.8-slim-bookworm

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Use bash as the default shell
SHELL ["bash", "-c"]

# Set the working directory
WORKDIR /app

# Add label descriptions
LABEL org.opencontainers.image.title="nubison-model" \
      org.opencontainers.image.description="A container image for nubison-model."\
      org.opencontainers.image.source="https://github.com/nubison/nubison-model"

COPY start_server.sh /app/start_server.sh
RUN chmod +x /app/start_server.sh

# Set environment variables with default values
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_URI=""
ENV DEBUG=""
ENV PORT=3000
ENV NUM_WORKERS=4

# Expose the port specified by the PORT environment variable
EXPOSE ${PORT}

RUN pip install uv==0.5.10

# Define the command to run your application
ENTRYPOINT [ "/app/start_server.sh" ]