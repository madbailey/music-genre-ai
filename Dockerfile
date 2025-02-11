#Base image
FROM tensorflow/tensorflow:2.13.0-gpu


WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "protobuf<4.0.0"

COPY . .

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV CUDA_VERSION 11.8.0
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64

COPY --chown=appuser:appuser . .

# Add CUDA to PATH
ENV PATH /usr/local/cuda-11.8/bin:$PATH

CMD ["python", "--version"]

