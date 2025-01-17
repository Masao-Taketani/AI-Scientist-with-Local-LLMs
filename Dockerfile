FROM continuumio/anaconda3:latest

RUN apt-get update && apt-get install -y \
    texlive-full \
    build-essential \
    screen

RUN curl -fsSL https://ollama.com/install.sh | sh
    
RUN pip install torch \
                transformers==4.47.1 \
                accelerate \
                datasets \
                backoff \
                autoawq \
                pyalex \
                aider-chat \
                pypdf \
                pymupdf4llm \
                matplotlib \
                jupyter notebook

RUN mkdir /work && chmod 777 /work
RUN conda init
WORKDIR /work