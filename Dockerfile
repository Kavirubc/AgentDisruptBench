FROM python:3.10-slim

LABEL maintainer="AgentDisruptBench Contributors"
LABEL description="Reproducible evaluation environment for AgentDisruptBench"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git make && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE Makefile ./
COPY python/ python/
COPY evaluation/ evaluation/
COPY tests/ tests/
COPY config/ config/
COPY examples/ examples/

# Install package
RUN pip install --no-cache-dir -e ".[dev,all,cli]"

# Default command
CMD ["make", "test"]
