FROM continuumio/anaconda3:latest

RUN apt-get update && apt-get install -y \
    texlive-full \
    build-essential
    
RUN pip install torch \
                transformers \
                accelerate \
                datasets \
                backoff \
                autoawq \
                jupyter notebook

RUN mkdir /work && chmod 777 /work
RUN conda init
WORKDIR /work