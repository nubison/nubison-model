FROM python:3.12.8-slim-bookworm

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Use bash as the default shell
SHELL ["bash", "-c"]

# Set the working directory
WORKDIR /app

# Copy the entrypoint script into the container
COPY start_server.sh /app/start_server.sh
RUN chmod +x /app/start_server.sh

# Set environment variables with default values
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_URI=""
ENV DEBUG=""
ENV PORT=3000

# Expose the port specified by the PORT environment variable
EXPOSE ${PORT}

RUN pip install uv==0.5.10

# Define the command to run your application
ENTRYPOINT [ "/app/start_server.sh" ]