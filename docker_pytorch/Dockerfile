# 1. 直接換成最新的 CUDA 13.0 開發版鏡像（全面解鎖 Blackwell 架構底層）
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# 2. 透過 deadsnakes PPA 精準安裝 Python 3.11 穩定正式版與 OpenCV 底層庫
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 強制將系統預設的 python 連結指向穩定版 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/python3.11 /usr/bin/python

# 3. 升級 pip 工具組
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 4. 【調整順序】優先安裝支援 CUDA 13 / Blackwell 的 PyTorch 預覽版
# 這樣可以確保後續安裝 requirements.txt 時，其他套件會自動依賴這個正確的 torch 版本
RUN python3 -m pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 5. 安裝一般週邊套件（讀取你的 requirements.txt）
# 💡 提示：請把 requirements.txt 裡面的 torch, torchvision 刪除或註解掉，避免重複安裝
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .