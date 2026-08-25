FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    libfftw3-dev \
    libopencv-dev \
    ninja-build \
    nlohmann-json3-dev \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY . .
RUN chmod +x ./scripts/*.sh \
 && cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DFAST_DEBLUR_NATIVE=OFF \
 && cmake --build build --parallel

CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
