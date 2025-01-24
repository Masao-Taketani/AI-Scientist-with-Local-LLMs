FROM continuumio/anaconda3:latest

RUN apt-get update && apt-get install -y \
    texlive-full \
    build-essential \
    screen \
    git

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN pip install torch
RUN pip install aider-chat \
                autoawq \
                backoff \
                openai \
                matplotlib \
                pypdf \
                pymupdf4llm \
                transformers \
                accelerate \
                datasets \
                pyalex \
                jupyter notebook

RUN mkdir /work && chmod 777 /work
RUN conda init
WORKDIR /work