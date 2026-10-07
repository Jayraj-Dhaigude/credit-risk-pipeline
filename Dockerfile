FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# LightGBM needs the OpenMP runtime, which the slim image does not include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install the libraries first: Docker caches this step, so rebuilds are fast
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# then the code and the production model
COPY src/ src/
COPY models/production/ models/production/

# run as a normal user, not as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]