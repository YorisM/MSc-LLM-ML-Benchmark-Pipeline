# 1. Pick a minimal, official Python runtime
FROM python:3.13-slim

# 2. Create a non‑root user for isolation
RUN useradd --create-home --shell /bin/bash sandboxuser

# 3. Set your working directory inside the container
WORKDIR /workspace

# 4. Copy only requirements first (caching!)
COPY Docker_requirements.txt /workspace/

# 5. Install your Python deps, clean apt afterwards
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       build-essential  \
 && pip install --no-cache-dir -r Docker_requirements.txt \
 && apt-get purge -y --auto-remove build-essential \
 && rm -rf /var/lib/apt/lists/*

# 6. Switch to the unprivileged user
USER sandboxuser

# 7. Default entrypoint: we’ll pass scripts directly
ENTRYPOINT ["python"]