# llama.cpp built from source against TheRock ROCm for gfx1151 (Strix Halo).
# Tight image — inference only, no Python, no training, no AITER, no custom forks.
FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Build + runtime system deps. No Python (llama.cpp is pure C++).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      build-essential cmake ninja-build \
      aria2 tar xz-utils \
      libatomic1 libnuma-dev libgomp1 libelf1t64 \
      libdrm-dev zlib1g-dev libssl-dev libcurl4-openssl-dev \
      libgoogle-perftools4 \
      procps \
    && rm -rf /var/lib/apt/lists/*

# 2. TheRock ROCm SDK → /opt/rocm (OS-agnostic tarball, resolves latest).
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
COPY scripts/install_rocm_sdk.sh /tmp/install_rocm_sdk.sh
RUN chmod +x /tmp/install_rocm_sdk.sh && \
    ROCM_MAJOR_VER=${ROCM_MAJOR_VER} GFX=${GFX} /tmp/install_rocm_sdk.sh && \
    rm /tmp/install_rocm_sdk.sh

# 3. ROCm env for the HIP compiler used by the llama.cpp build.
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    HIPCXX=/opt/rocm/llvm/bin/clang++ \
    CMAKE_PREFIX_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib

# 4. llama.cpp from a pinned commit (override with --build-arg LLAMA_COMMIT=<sha> or LLAMA_COMMIT="" for HEAD).
ARG LLAMA_COMMIT=15fa3c493
RUN git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
WORKDIR /opt/llama.cpp
RUN if [ -n "$LLAMA_COMMIT" ]; then \
      git checkout "$LLAMA_COMMIT"; \
    else \
      echo "Tracking llama.cpp HEAD: $(git rev-parse --short HEAD)"; \
    fi

# 5. Build with HIP backend + rocWMMA flash-attention for RDNA3.5 perf.
# GGML_HIP_ROCWMMA_FATTN=ON enables the fused flash-attention kernels that
# are a meaningful speedup on Strix Halo; rocWMMA headers ship with TheRock.
RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
    echo "bitcode: $HIP_DEVICE_LIB_PATH" && \
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP=ON \
      -DGGML_HIP_ROCWMMA_FATTN=ON \
      -DGPU_TARGETS=gfx1151 \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build -j 4 && \
    # proper install to /usr/local so ldconfig finds libllama-common, libmtmd, etc.
    cmake --install build && \
    ldconfig && \
    rm -rf /opt/llama.cpp/build

ENV PATH=/usr/local/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    HSA_OVERRIDE_GFX_VERSION=11.5.1

WORKDIR /models
CMD ["/bin/bash"]
