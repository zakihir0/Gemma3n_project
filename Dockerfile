# Stage 1: ベースイメージの選定
# vLLMとgemma3n.py内のPyTorch(cu124)の要件に合うCUDA 12.4.1をベースにする
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Stage 2: 環境変数の設定
# ビルド中の対話を防ぎ、タイムゾーンを標準化
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV SHELL=/bin/bash

# Stage 3: Paperspaceの必須要件である作業ディレクトリの設定
WORKDIR /notebooks

# Stage 4: システム依存関係のインストール
# Python, pip, gitなどをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    git \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Stage 5: プロジェクトファイルのコピー
# 現在のディレクトリのすべてのファイルをコンテナにコピー
COPY . .

# Stage 6: Pythonライブラリのインストール
# vLLM, Transformers, およびgemma3n.pyに必要なすべてのライブラリをインストール
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install vllm && \
    pip3 install unsloth faiss-cpu sentence-transformers wikipedia Pillow pyarrow datasets && \
    pip3 install --no-deps git+https://github.com/huggingface/transformers.git && \
    pip3 install --no-deps --upgrade timm && \
    pip3 install jupyterlab

# Stage 7: Paperspaceの必須要件であるポートの開放
EXPOSE 8888

# Stage 8: Paperspaceの必須要件であるJupyterLabの起動コマンド
# このコマンドにより、Paperspaceがコンテナ内のJupyterサーバーに接続できる
CMD ["jupyter", "lab", "--port=8888", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--LabApp.trust_xheaders=True", "--LabApp.disable_check_xsrf=False", "--LabApp.allow_remote_access=True", "--LabApp.allow_origin='*'" ]