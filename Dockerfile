# Base image: Ubuntu 22.04
FROM ubuntu:22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl unzip build-essential cmake libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
