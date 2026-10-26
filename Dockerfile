FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TABPFN_MODEL_CACHE_DIR=/models \
    MERGEMETER_CACHE=/tmp/mergemeter-cache \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app
# CPU-only torch keeps the image ~1 GB smaller than the default CUDA wheel.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the open TabPFN v2 classifier checkpoint into the image (plain HTTP, no token).
RUN python -c "\
from tabpfn import TabPFNClassifier; from tabpfn.constants import ModelVersion; import numpy as np; \
TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu', n_estimators=1).fit(np.random.rand(10, 2), np.arange(10) % 2)" \
 && chmod -R a+rX /models

COPY . .
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD ["python", "app.py"]
