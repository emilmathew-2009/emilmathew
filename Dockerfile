FROM python:3.12-slim
ARG AUDIVERIS_VERSION=5.11.0
ENV AUDIVERIS_VERSION=${AUDIVERIS_VERSION} OMR_WORKDIR=/tmp/emil-omr PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless tesseract-ocr wget ca-certificates \
    libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libxtst6 libxi6 \
    libfreetype6 fonts-dejavu ffmpeg && rm -rf /var/lib/apt/lists/*
RUN wget -q -O /tmp/audiveris.deb https://github.com/Audiveris/audiveris/releases/download/${AUDIVERIS_VERSION}/Audiveris-${AUDIVERIS_VERSION}-ubuntu24.04-x86_64.deb \
    && dpkg -i /tmp/audiveris.deb || (apt-get update && apt-get -f install -y && rm -rf /var/lib/apt/lists/*) \
    && rm -f /tmp/audiveris.deb
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY omr-service.py /app/omr-service.py
COPY audio_transcriber.py /app/audio_transcriber.py
EXPOSE 8000
CMD ["uvicorn","omr-service:app","--host","0.0.0.0","--port","8000"]
