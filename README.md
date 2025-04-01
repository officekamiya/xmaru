# OS環境の構築

## Ubuntuの画面スケールの追加

    gsettings set org.gnome.mutter experimental-features "['scale-monitor-framebuffer']"

[設定] -> [ディスプレイ]（または [設定] -> [アピアランス]）を開くと、Fractional Scaling をオンにできるトグルや、125% / 150% などの追加スケール項目が表示される。

## ZSH

ZSHのインストール

    sudo apt install zsh

ZSHの適用

    chsh -s /bin/zsh

## GIT

GITのインストール

    sudo apt install git

# GPU（NVIDIA関連の設定）

## GPU Driverの確認

    nvidia-smi

## CUDAのインストール

CUDAリポジトリのダウンロード

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

CUDAリポジトリのインストール

    sudo dpkg -i cuda-keyring_1.1-1_all.deb

CUDAのインストール

    sudo apt update
    sudo apt install cuda-toolkit

# CUDA PATH設定

.zshrcの修正

    vi ~/.zshrc

内容

    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

.zshrcの適用

    source ~/.zshrc

nvccのパスの確認

    nvcc --version

# Python

Ubuntuのプリインストール版のPythonには、venvを作成する機能がないため、aptで別にインストールする必要があり。

venvのインストール

    sudo apt-get install python3.12-venv

仮想環境の設定

    python3 -m venv .venv

仮想環境の有効化

    source .venv/bin/activate

pipのアップデート

    pip install --upgrade pip

PyTorchのインストール

    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Langchain

pip install langchain

gemini用

    pip install google-generativeai
    pip install langchain-google-genai

gemma用

    pip install langchain-google-vertexai

GemmaLocalHF用

    sudo apt-get install git-lfs
    git lfs install
    

huggingfaceログイン用

    pip install huggingface_hub
    huggingface-cli login

プロジェクト/models

    git lfs clone https://huggingface.co/google/gemma-3-1b-it
    git lfs clone https://huggingface.co/google/gemma-3-1b-pt

https://aistudio.google.com/apikey

